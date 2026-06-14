"""MacRAG context assembly: scale-up, neighbor expansion, and merging.

Port of the post-rerank logic in `third_party/MacRAG/MacRAG/src/main_macrag.py`
(`sort_section` chunk_ext expansion + merge): take the top-k2 reranked parent
chunks, expand each by `chunk_ext` neighbor hops, then merge adjacent chunks
within the same document, de-duplicating the chunk overlap
(`combine_without_overlap`).

Harness adaptations (documented in the trace): the document unit is the
act+scene (single-document corpus); merged blocks carry exact token counts
from the chunks' start/end token offsets; the final context enforces the
token budget by dropping whole blocks. merge_version 1 produces one block
per scene; merge_version 2 produces one block per contiguous chunk run.
"""

from __future__ import annotations

from typing import Any

from hamlet_qa.core.context import (
    ContextAssemblyRequest,
    ContextAssemblyResult,
    make_pseudo_chunk,
)

MACRAG_DEVIATIONS = [
    "local reader replaces GPT-4o as the offline summarizer",
    "harness 256-token chunks reused as parents (official re-chunks at "
    "1500/500 chars)",
    "Qwen embedder/reranker replace multilingual-e5 / ms-marco-MiniLM",
    "document unit for expansion/merging is the act+scene",
    "R&B generation strategy only; token budget enforced by dropping blocks",
]


def combine_without_overlap(text1: str, text2: str) -> str:
    """Official overlap-aware concatenation from main_macrag.py."""
    if not text1:
        return text2
    max_overlap = min(len(text1), len(text2))
    overlap = 0
    for length in range(max_overlap, 0, -1):
        if text1.endswith(text2[:length]):
            overlap = length
            break
    return text1 + text2[overlap:]


def assemble_macrag(request: ContextAssemblyRequest) -> ContextAssemblyResult:
    if not request.retrieval_trace:
        raise ValueError("macrag requires a macrag retrieval trace")

    params = request.feature_params
    top_k2 = int(params.get("macrag_top_k2", 7))
    chunk_ext = int(params.get("macrag_chunk_ext", 1))
    merge_version = int(params.get("macrag_merge_version", 1))

    chunk_lookup = request.chunk_lookup
    by_global_index = {
        int(chunk["global_index"]): chunk for chunk in chunk_lookup.values()
    }
    original_hit_chunk_ids = [str(row["chunk_id"]) for row in request.retrieval_trace]

    top_rows = [
        row
        for row in request.retrieval_trace[:top_k2]
        if str(row["chunk_id"]) in chunk_lookup
    ]
    rank_by_chunk: dict[str, int] = {}

    # chunk_ext neighbor expansion within the same scene.
    expanded: dict[str, dict[str, Any]] = {}
    for row in top_rows:
        seed = chunk_lookup[str(row["chunk_id"])]
        seed_scene = str(seed["scene_id"])
        seed_index = int(seed["global_index"])
        for offset in range(-chunk_ext, chunk_ext + 1):
            neighbor = by_global_index.get(seed_index + offset)
            if neighbor is None or str(neighbor["scene_id"]) != seed_scene:
                continue
            neighbor_id = str(neighbor["chunk_id"])
            expanded.setdefault(neighbor_id, neighbor)
            rank = int(row["rank"])
            if neighbor_id not in rank_by_chunk or rank < rank_by_chunk[neighbor_id]:
                rank_by_chunk[neighbor_id] = rank

    # Group expanded chunks into contiguous runs per scene.
    chunks_sorted = sorted(
        expanded.values(), key=lambda chunk: int(chunk["global_index"])
    )
    runs: list[list[dict[str, Any]]] = []
    for chunk in chunks_sorted:
        if (
            runs
            and int(chunk["global_index"]) == int(runs[-1][-1]["global_index"]) + 1
            and str(chunk["scene_id"]) == str(runs[-1][-1]["scene_id"])
        ):
            runs[-1].append(chunk)
        else:
            runs.append([chunk])

    if merge_version == 1:
        # One block per scene: concatenate that scene's runs.
        groups: dict[str, list[list[dict[str, Any]]]] = {}
        for run in runs:
            groups.setdefault(str(run[0]["scene_id"]), []).append(run)
        blocks = [
            {"runs": scene_runs}
            for scene_runs in groups.values()
        ]
    else:
        blocks = [{"runs": [run]} for run in runs]

    def block_chunk_ids(block: dict[str, Any]) -> list[str]:
        return [
            str(chunk["chunk_id"]) for run in block["runs"] for chunk in run
        ]

    def block_rank(block: dict[str, Any]) -> int:
        return min(rank_by_chunk[chunk_id] for chunk_id in block_chunk_ids(block))

    def block_tokens(block: dict[str, Any]) -> int:
        total = 0
        for run in block["runs"]:
            total += int(run[-1]["end_token"]) - int(run[0]["start_token"])
        return total

    def block_to_chunk(block: dict[str, Any]) -> dict[str, Any]:
        member_ids = block_chunk_ids(block)
        if len(member_ids) == 1:
            return dict(chunk_lookup[member_ids[0]])
        run_texts: list[str] = []
        for run in block["runs"]:
            merged = ""
            for chunk in run:
                merged = combine_without_overlap(merged, str(chunk["text"]))
            run_texts.append(merged)
        first = block["runs"][0][0]
        last = block["runs"][-1][-1]
        pseudo = make_pseudo_chunk(
            chunk_id=(
                f"macrag_merged_{first['chunk_id']}_to_{last['chunk_id']}"
            ),
            text="\n".join(run_texts),
            scene_title=str(first["scene_title"]),
            scene_id=str(first["scene_id"]),
        )
        # Exact token accounting from the chunk offsets, not word count.
        pseudo["act"] = first["act"]
        pseudo["scene"] = first["scene"]
        pseudo["global_index"] = first["global_index"]
        pseudo["start_token"] = first["start_token"]
        pseudo["end_token"] = last["end_token"]
        pseudo["token_count"] = block_tokens(block)
        pseudo["member_chunk_ids"] = member_ids
        return pseudo

    ordered_blocks = sorted(blocks, key=block_rank)
    selected_chunks: list[dict[str, Any]] = []
    selected_ids: list[str] = []
    dropped_over_budget: list[list[str]] = []
    total_tokens = 0
    for block in ordered_blocks:
        tokens = block_tokens(block)
        if total_tokens + tokens > request.context_budget:
            dropped_over_budget.append(block_chunk_ids(block))
            continue
        chunk = block_to_chunk(block)
        selected_chunks.append(chunk)
        selected_ids.append(str(chunk["chunk_id"]))
        total_tokens += tokens

    trace = {
        "method": "macrag",
        "source": "third_party/MacRAG sort_section chunk_ext + merge port",
        "top_k2": top_k2,
        "chunk_ext": chunk_ext,
        "merge_version": merge_version,
        "seed_chunk_ids": [str(row["chunk_id"]) for row in top_rows],
        "expanded_chunk_ids": sorted(
            expanded,
            key=lambda chunk_id: int(chunk_lookup[chunk_id]["global_index"]),
        ),
        "blocks": [
            {
                "chunk_ids": block_chunk_ids(block),
                "rank": block_rank(block),
                "tokens": block_tokens(block),
            }
            for block in ordered_blocks
        ],
        "dropped_over_budget_blocks": dropped_over_budget,
        "deviations": list(MACRAG_DEVIATIONS),
    }
    retrieval_method = str(
        request.retrieval_trace[0].get("retrieval_method", "macrag")
    )
    return ContextAssemblyResult(
        selected_chunk_ids=selected_ids,
        selected_chunks=selected_chunks,
        original_hit_chunk_ids=original_hit_chunk_ids,
        retrieval_trace=[dict(row) for row in request.retrieval_trace],
        retrieval_method=retrieval_method,
        prompt_order="macrag_block_rank",
        context_assembly_trace=trace,
    )
