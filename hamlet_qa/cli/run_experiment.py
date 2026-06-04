"""CLI for running Hamlet QA treatments."""

from __future__ import annotations

import argparse

from hamlet_qa.core.config import (
    DEFAULT_BM25_B,
    DEFAULT_BM25_K1,
    DEFAULT_CONTEXT_ASSEMBLY_CACHE_DIR,
    DEFAULT_CONTEXT_BUDGETS,
    DEFAULT_DOMAIN_KG_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GPU_LAYOUT,
    DEFAULT_READER_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SETR_MAX_PASSAGES,
    DEFAULT_SETR_SELECTOR_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TREATMENTS,
    GPU_LAYOUTS,
    RunConfig,
)
from hamlet_qa.core.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Hamlet QA failure-analysis treatments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--document", default="data/hamlet.txt")
    parser.add_argument("--chunks", default="data/hamlet_chunks.jsonl")
    parser.add_argument("--questions", default="data/hamlet_questions.json")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-name", default="hamlet_probe")
    parser.add_argument("--reader-model", default=DEFAULT_READER_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help="Cross-encoder reranker model. Use 'none' to disable reranking.",
    )
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
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--gpu-layout",
        choices=sorted(GPU_LAYOUTS),
        default=DEFAULT_GPU_LAYOUT,
        help=(
            "Device placement preset. 'single' keeps the default staged "
            "single-GPU run intact; 'a40-2gpu' uses cuda:0 for retrieval and "
            "reranking and cuda:1 for the vLLM reader; 'a40-3gpu' uses cuda:0 "
            "for the embedder, cuda:1 for the reranker, and cuda:2 for the "
            "vLLM reader."
        ),
    )
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument(
        "--embedding-device",
        default=None,
        help="Override the embedder device selected by --gpu-layout.",
    )
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument(
        "--reranker-device",
        default=None,
        help="Override the reranker device selected by --gpu-layout.",
    )
    parser.add_argument(
        "--reader-device",
        default=None,
        help="Override the vLLM reader device selected by --gpu-layout.",
    )
    parser.add_argument("--bm25-k1", type=float, default=DEFAULT_BM25_K1)
    parser.add_argument("--bm25-b", type=float, default=DEFAULT_BM25_B)
    parser.add_argument(
        "--domain-kg",
        default=DEFAULT_DOMAIN_KG_PATH,
        help="Editable domain knowledge graph used by the domain treatment.",
    )
    parser.add_argument(
        "--context-assembly-cache-dir",
        default=DEFAULT_CONTEXT_ASSEMBLY_CACHE_DIR,
        help="Directory for cached SetR selector prompts, outputs, and parsed selections.",
    )
    parser.add_argument(
        "--setr-max-passages",
        type=int,
        default=DEFAULT_SETR_MAX_PASSAGES,
        help="Number of dense candidates exposed to the SetR selector.",
    )
    parser.add_argument(
        "--setr-selector-max-tokens",
        type=int,
        default=DEFAULT_SETR_SELECTOR_MAX_TOKENS,
        help="Maximum tokens for the SetR selection_IRI selector response.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> RunConfig:
    gpu_layout = GPU_LAYOUTS[args.gpu_layout]
    embedding_device = args.embedding_device or gpu_layout["embedding_device"]
    reranker_device = args.reranker_device or gpu_layout["reranker_device"]
    reader_device = args.reader_device or gpu_layout["reader_device"]
    return RunConfig(
        document_path=args.document,
        chunks_path=args.chunks,
        questions_path=args.questions,
        output_dir=args.output_dir,
        run_name=args.run_name,
        reader_model=args.reader_model,
        embedding_model=args.embedding_model,
        reranker_model=None
        if str(args.reranker_model).lower() in {"", "none", "null", "off", "false"}
        else args.reranker_model,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        context_budgets=args.context_budgets,
        treatments=args.treatments,
        top_k=args.top_k,
        random_seed=args.random_seed,
        gpu_layout=args.gpu_layout,
        embedding_batch_size=args.embedding_batch_size,
        embedding_device=embedding_device,
        reranker_batch_size=args.reranker_batch_size,
        reranker_device=reranker_device,
        reader_device=reader_device,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        domain_kg_path=args.domain_kg,
        context_assembly_cache_dir=args.context_assembly_cache_dir,
        setr_max_passages=args.setr_max_passages,
        setr_selector_max_tokens=args.setr_selector_max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        prepare_only=args.prepare_only,
        overwrite=args.overwrite,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    results_path = run_experiment(config)
    print(f"Wrote results to {results_path}")


if __name__ == "__main__":
    main()
