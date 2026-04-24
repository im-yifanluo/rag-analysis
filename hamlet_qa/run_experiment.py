"""CLI for running Hamlet QA treatments."""

from __future__ import annotations

import argparse

from hamlet_qa.config import (
    DEFAULT_CONTEXT_BUDGETS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_READER_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_NEIGHBOR_WINDOW,
    DEFAULT_TOP_K,
    DEFAULT_TREATMENTS,
    RunConfig,
)
from hamlet_qa.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Hamlet QA failure-analysis treatments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--document", default="hamlet.txt")
    parser.add_argument("--chunks", default="data/hamlet_chunks.jsonl")
    parser.add_argument("--questions", default="data/hamlet_questions.json")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-name", default="hamlet_probe")
    parser.add_argument("--reader-model", default=DEFAULT_READER_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--context-budgets",
        type=int,
        nargs="+",
        default=DEFAULT_CONTEXT_BUDGETS,
    )
    parser.add_argument(
        "--treatments",
        nargs="+",
        default=DEFAULT_TREATMENTS,
        choices=DEFAULT_TREATMENTS,
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--neighbor-window", type=int, default=DEFAULT_NEIGHBOR_WINDOW)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(
        document_path=args.document,
        chunks_path=args.chunks,
        questions_path=args.questions,
        output_dir=args.output_dir,
        run_name=args.run_name,
        reader_model=args.reader_model,
        embedding_model=args.embedding_model,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        context_budgets=args.context_budgets,
        treatments=args.treatments,
        top_k=args.top_k,
        neighbor_window=args.neighbor_window,
        embedding_batch_size=args.embedding_batch_size,
        embedding_device=args.embedding_device,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        prepare_only=args.prepare_only,
        overwrite=args.overwrite,
    )
    results_path = run_experiment(config)
    print(f"Wrote results to {results_path}")


if __name__ == "__main__":
    main()
