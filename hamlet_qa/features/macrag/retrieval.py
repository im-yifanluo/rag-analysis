"""MacRAG query-time retrieval over summary slices.

Mirrors the official query path (`main_macrag.py`): embed the query, take
top-k1 summary slices from a flat inner-product index, map slices to their
parent chunks with keep-first dedupe, then rerank parent chunks with the
cross-encoder. The harness embedder/reranker (Qwen) replace the official
multilingual-e5 / ms-marco-MiniLM models (documented deviation).
"""

from __future__ import annotations

from typing import Any

from hamlet_qa.features.macrag.index import load_macrag_artifacts


def slice_hits_to_parent_candidates(
    slice_hits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Dedupe slice hits by parent chunk, keeping the best slice per parent."""
    candidates: list[dict[str, Any]] = []
    seen_parents: set[str] = set()
    for hit in slice_hits:
        parent_chunk_id = str(hit["parent_chunk_id"])
        if parent_chunk_id in seen_parents:
            continue
        seen_parents.add(parent_chunk_id)
        candidates.append(dict(hit))
    return candidates


def build_macrag_traces(
    config: Any,
    chunks: list[dict[str, Any]],
    questions: list[Any],
    traces: dict[str, dict[str, list[dict[str, Any]]]],
    feature_handles: dict[str, Any] | None = None,
) -> None:
    """Populate traces[qid]["macrag"] with reranked parent-chunk rows."""
    handles = feature_handles or {}
    stub_retriever = handles.get("macrag_retriever")
    if stub_retriever is not None:
        for question in questions:
            traces[question.id]["macrag"] = stub_retriever.retrieve(
                question.question,
                config.macrag_top_k1,
            )
        return

    import numpy as np

    from hamlet_qa.core.experiment import clear_cuda_cache, make_reranker
    from hamlet_qa.core.retrieval import SentenceTransformerEmbedder

    artifacts = handles.get("macrag_artifacts") or load_macrag_artifacts(
        config.macrag_artifacts_dir
    )
    slices = artifacts["slices"]
    chunk_lookup = {str(chunk["chunk_id"]): chunk for chunk in chunks}

    # Stage 1: embedder — slice index + query search.
    import faiss

    embedder = SentenceTransformerEmbedder(
        config.embedding_model,
        device=config.embedding_device,
        batch_size=config.embedding_batch_size,
    )
    slice_hits_per_question: dict[str, list[dict[str, Any]]] = {}
    try:
        embeddings = embedder.embed_passages([str(item["text"]) for item in slices])
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))
        k = min(config.macrag_top_k1, len(slices))
        for question in questions:
            query_embedding = embedder.embed_query(question.question).reshape(1, -1)
            scores, indices = index.search(query_embedding.astype(np.float32), k)
            hits = []
            for slice_rank, (score, slice_index) in enumerate(
                zip(scores[0], indices[0]), start=1
            ):
                if slice_index < 0:
                    continue
                record = slices[int(slice_index)]
                hits.append(
                    {
                        "slice_id": record["slice_id"],
                        "parent_chunk_id": str(record["parent_chunk_id"]),
                        "slice_rank": slice_rank,
                        "slice_score": float(score),
                        "slice_kind": record.get("slice_kind"),
                    }
                )
            slice_hits_per_question[question.id] = hits
    finally:
        del embedder
        clear_cuda_cache()

    # Stage 2: reranker — score parent chunks against the question.
    reranker = None
    if config.reranker_model:
        reranker = make_reranker(config)
    try:
        for question in questions:
            candidates = slice_hits_to_parent_candidates(
                slice_hits_per_question.get(question.id, [])
            )
            candidates = [
                candidate
                for candidate in candidates
                if candidate["parent_chunk_id"] in chunk_lookup
            ]
            if reranker is not None:
                documents = [
                    str(chunk_lookup[candidate["parent_chunk_id"]]["text"])
                    for candidate in candidates
                ]
                rerank_scores = reranker.score(question.question, documents)
                for candidate, rerank_score in zip(candidates, rerank_scores):
                    candidate["rerank_score"] = float(rerank_score)
                candidates.sort(
                    key=lambda candidate: (
                        -float(candidate["rerank_score"]),
                        int(candidate["slice_rank"]),
                    )
                )
            rows: list[dict[str, Any]] = []
            for rank, candidate in enumerate(candidates, start=1):
                chunk = chunk_lookup[candidate["parent_chunk_id"]]
                rows.append(
                    {
                        "chunk_id": str(chunk["chunk_id"]),
                        "rank": rank,
                        "score": candidate.get(
                            "rerank_score", candidate["slice_score"]
                        ),
                        "rerank_score": candidate.get("rerank_score"),
                        "slice_id": candidate["slice_id"],
                        "slice_rank": candidate["slice_rank"],
                        "slice_score": candidate["slice_score"],
                        "global_index": chunk["global_index"],
                        "act": chunk["act"],
                        "scene": chunk["scene"],
                        "scene_title": chunk["scene_title"],
                        "retrieval_method": "macrag_summary_slices_reranked"
                        if reranker is not None
                        else "macrag_summary_slices",
                    }
                )
            traces[question.id]["macrag"] = rows
    finally:
        if reranker is not None:
            del reranker
            clear_cuda_cache()
