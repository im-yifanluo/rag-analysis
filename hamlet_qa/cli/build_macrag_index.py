"""Build the MacRAG hierarchical artifacts (chunk summaries + slices).

Offline step mirroring `gen_index_macrag.py` steps 2-3: summarize every
harness chunk with the reader model (official: GPT-4o), slice each summary
into 450/300-char pieces, and write JSONL artifacts that the `macrag`
treatment retrieves over at run time. Summaries are cached per chunk so
reruns are incremental.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hamlet_qa.core.config import (
    DEFAULT_MACRAG_ARTIFACTS_DIR,
    DEFAULT_MACRAG_SLICE_OVERLAP,
    DEFAULT_MACRAG_SLICE_SIZE,
    DEFAULT_READER_MODEL,
)
from hamlet_qa.core.io import load_jsonl
from hamlet_qa.core.llm_cache import JsonKVCache, stable_hash
from hamlet_qa.features.macrag.index import (
    SLICES_FILENAME,
    SUMMARIES_FILENAME,
    build_slices_for_chunk,
)
from hamlet_qa.features.macrag.summarize import summarize_chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MacRAG summary and slice artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--chunks", default="data/hamlet_chunks.jsonl")
    parser.add_argument("--output-dir", default=DEFAULT_MACRAG_ARTIFACTS_DIR)
    parser.add_argument("--summarizer-model", default=DEFAULT_READER_MODEL)
    parser.add_argument("--slice-size", type=int, default=DEFAULT_MACRAG_SLICE_SIZE)
    parser.add_argument(
        "--slice-overlap", type=int, default=DEFAULT_MACRAG_SLICE_OVERLAP
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = load_jsonl(args.chunks)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = JsonKVCache(output_dir / "summary_cache.json", section="macrag_summaries")

    pending = []
    summaries: dict[str, dict] = {}
    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        cache_key = stable_hash(
            {
                "chunk_id": chunk_id,
                "text": chunk["text"],
                "model": args.summarizer_model,
            }
        )
        cached = cache.get(cache_key)
        if cached is not None:
            summaries[chunk_id] = cached
        else:
            pending.append((chunk, cache_key))

    if pending:
        from hamlet_qa.core.generation import VLLMReader

        reader = VLLMReader(
            args.summarizer_model,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            device=args.device,
        )
        for index, (chunk, cache_key) in enumerate(pending, start=1):
            chunk_id = str(chunk["chunk_id"])
            record = summarize_chunk(str(chunk["text"]), reader)
            record["summary_model"] = args.summarizer_model
            summaries[chunk_id] = record
            cache.set(cache_key, record)
            if index % 25 == 0 or index == len(pending):
                cache.save()
                print(f"Summarized {index}/{len(pending)} chunks")
    else:
        print("All chunk summaries served from cache.")

    summary_rows = []
    slice_rows = []
    fallback_count = 0
    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        record = summaries[chunk_id]
        if record.get("fallback"):
            fallback_count += 1
        summary_rows.append(
            {
                "chunk_id": chunk_id,
                "title": record.get("title", ""),
                "keywords": record.get("keywords", ""),
                "subheadings": record.get("subheadings", ""),
                "summary": record.get("summary", ""),
                "summary_model": record.get("summary_model", args.summarizer_model),
                "fallback": bool(record.get("fallback", False)),
            }
        )
        slice_rows.extend(
            build_slices_for_chunk(
                chunk_id,
                record,
                slice_size=args.slice_size,
                slice_overlap=args.slice_overlap,
            )
        )

    with (output_dir / SUMMARIES_FILENAME).open("w", encoding="utf-8") as handle:
        for row in summary_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / SLICES_FILENAME).open("w", encoding="utf-8") as handle:
        for row in slice_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"Wrote {len(summary_rows)} summaries ({fallback_count} fallbacks) and "
        f"{len(slice_rows)} slices to {output_dir}"
    )


if __name__ == "__main__":
    main()
