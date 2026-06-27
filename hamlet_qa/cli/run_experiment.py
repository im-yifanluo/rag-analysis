"""CLI for running Hamlet QA treatments."""

from __future__ import annotations

import argparse

from hamlet_qa.core.config import (
    DEFAULT_BM25_B,
    DEFAULT_BM25_K1,
    DEFAULT_CONTEXT_ASSEMBLY_CACHE_DIR,
    DEFAULT_CONTEXT_BUDGETS,
    DEFAULT_CRAG_DECOMPOSE_MODE,
    DEFAULT_CRAG_EXTERNAL_TOP_K,
    DEFAULT_CRAG_LOWER_THRESHOLD,
    DEFAULT_CRAG_NDOCS,
    DEFAULT_CRAG_UPPER_THRESHOLD,
    DEFAULT_DOMAIN_KG_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GPU_LAYOUT,
    DEFAULT_MACRAG_ARTIFACTS_DIR,
    DEFAULT_MACRAG_CHUNK_EXT,
    DEFAULT_MACRAG_MERGE_VERSION,
    DEFAULT_MACRAG_TOP_K1,
    DEFAULT_MACRAG_TOP_K2,
    DEFAULT_READER_MODEL,
    DEFAULT_RECOMP_ABSTRACTIVE_MODE,
    DEFAULT_RECOMP_ABSTRACTIVE_MODEL,
    DEFAULT_RECOMP_EXTRACTIVE_MODEL,
    DEFAULT_RECOMP_INPUT_DOCS,
    DEFAULT_RECOMP_TOP_SENTENCES,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SETR_MAX_PASSAGES,
    DEFAULT_SETR_SELECTOR_MAX_TOKENS,
    DEFAULT_SUPPORT_CANDIDATE_CHUNKS,
    DEFAULT_SUPPORT_INCLUDE_NEIGHBORS,
    DEFAULT_SUPPORT_MAX_NODES,
    DEFAULT_SUPPORT_MAX_SELECTED_UNITS,
    DEFAULT_SUPPORT_MAX_UNIT_TOKENS,
    DEFAULT_SUPPORT_MAX_UNITS_TOTAL,
    DEFAULT_SUPPORT_MIN_UNIT_SCORE,
    DEFAULT_SUPPORT_NEIGHBOR_HOPS,
    DEFAULT_SUPPORT_NODE_CANDIDATE_CATALOG_K,
    DEFAULT_SUPPORT_NODE_COVERAGE_THRESHOLD,
    DEFAULT_SUPPORT_NODE_INDUCTION_MAX_TOKENS,
    DEFAULT_SUPPORT_PROMPT_ORDER,
    DEFAULT_SUPPORT_REDUNDANCY_BETA,
    DEFAULT_SUPPORT_SCORE_CACHE_PATH,
    DEFAULT_SUPPORT_TEACHER_MAX_TOKENS,
    DEFAULT_SUPPORT_TEACHER_UNITS_PER_NODE,
    DEFAULT_SUPPORT_TOKEN_EXPONENT_TAU,
    DEFAULT_SUPPORT_UNIT_TYPES,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TREATMENTS,
    GPU_LAYOUTS,
    RunConfig,
)
from hamlet_qa.core.experiment import run_experiment
from hamlet_qa.features.registry import known_treatment_names


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
        choices=sorted(known_treatment_names()),
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
    parser.add_argument(
        "--crag-ndocs",
        type=int,
        default=DEFAULT_CRAG_NDOCS,
        help="Number of top dense candidates the CRAG evaluator judges.",
    )
    parser.add_argument(
        "--crag-upper-threshold",
        type=float,
        default=DEFAULT_CRAG_UPPER_THRESHOLD,
        help="Evaluator score at or above which a document counts as Correct.",
    )
    parser.add_argument(
        "--crag-lower-threshold",
        type=float,
        default=DEFAULT_CRAG_LOWER_THRESHOLD,
        help="Evaluator score at or above which a document counts as Ambiguous.",
    )
    parser.add_argument(
        "--crag-decompose-mode",
        choices=["fixed_num", "excerption", "selection"],
        default=DEFAULT_CRAG_DECOMPOSE_MODE,
        help="CRAG decompose-then-recompose strip mode (official modes).",
    )
    parser.add_argument(
        "--crag-external-top-k",
        type=int,
        default=DEFAULT_CRAG_EXTERNAL_TOP_K,
        help="BM25 hits retrieved by the doc-wide corrective re-retrieval.",
    )
    parser.add_argument(
        "--crag-evaluator-device",
        default=None,
        help=(
            "Device for the CRAG strip evaluator. Defaults to cpu on the "
            "'single' GPU layout and the reranker device otherwise."
        ),
    )
    parser.add_argument(
        "--macrag-artifacts-dir",
        default=DEFAULT_MACRAG_ARTIFACTS_DIR,
        help="Directory holding build_macrag_index outputs.",
    )
    parser.add_argument("--macrag-top-k1", type=int, default=DEFAULT_MACRAG_TOP_K1)
    parser.add_argument("--macrag-top-k2", type=int, default=DEFAULT_MACRAG_TOP_K2)
    parser.add_argument(
        "--macrag-chunk-ext",
        type=int,
        choices=[0, 1, 2],
        default=DEFAULT_MACRAG_CHUNK_EXT,
        help="MacRAG neighbor expansion hops within the same scene.",
    )
    parser.add_argument(
        "--macrag-merge-version",
        type=int,
        choices=[1, 2],
        default=DEFAULT_MACRAG_MERGE_VERSION,
    )
    parser.add_argument(
        "--recomp-extractive-model",
        default=DEFAULT_RECOMP_EXTRACTIVE_MODEL,
    )
    parser.add_argument(
        "--recomp-abstractive-model",
        default=DEFAULT_RECOMP_ABSTRACTIVE_MODEL,
    )
    parser.add_argument(
        "--recomp-abstractive-mode",
        choices=["t5", "prompted_qwen"],
        default=DEFAULT_RECOMP_ABSTRACTIVE_MODE,
        help=(
            "Abstractive compressor: original T5 checkpoint, or the reader "
            "model prompted with the RECOMP paper Table 8 prompt."
        ),
    )
    parser.add_argument(
        "--recomp-input-docs",
        type=int,
        default=DEFAULT_RECOMP_INPUT_DOCS,
        help="Top dense candidates passed to the RECOMP compressors.",
    )
    parser.add_argument(
        "--recomp-top-sentences",
        type=int,
        default=DEFAULT_RECOMP_TOP_SENTENCES,
        help="Sentences kept by the extractive compressor.",
    )
    # reader_support (our method)
    parser.add_argument(
        "--support-candidate-chunks",
        type=int,
        default=DEFAULT_SUPPORT_CANDIDATE_CHUNKS,
        help="Top dense candidates that seed reader_support evidence units.",
    )
    parser.add_argument(
        "--support-node-candidate-catalog-k",
        type=int,
        default=DEFAULT_SUPPORT_NODE_CANDIDATE_CATALOG_K,
    )
    parser.add_argument(
        "--support-max-nodes", type=int, default=DEFAULT_SUPPORT_MAX_NODES
    )
    parser.add_argument(
        "--support-teacher-units-per-node",
        type=int,
        default=DEFAULT_SUPPORT_TEACHER_UNITS_PER_NODE,
        help="Units prefiltered per node before reader-teacher scoring.",
    )
    parser.add_argument(
        "--support-unit-types", default=DEFAULT_SUPPORT_UNIT_TYPES,
        help="Comma-separated unit types to construct.",
    )
    parser.add_argument(
        "--support-include-neighbors",
        type=lambda v: str(v).lower() not in {"0", "false", "no", "off"},
        default=DEFAULT_SUPPORT_INCLUDE_NEIGHBORS,
    )
    parser.add_argument(
        "--support-neighbor-hops", type=int, default=DEFAULT_SUPPORT_NEIGHBOR_HOPS
    )
    parser.add_argument(
        "--support-max-units-total", type=int, default=DEFAULT_SUPPORT_MAX_UNITS_TOTAL
    )
    parser.add_argument(
        "--support-max-unit-tokens", type=int, default=DEFAULT_SUPPORT_MAX_UNIT_TOKENS
    )
    parser.add_argument(
        "--support-node-coverage-threshold",
        type=float,
        default=DEFAULT_SUPPORT_NODE_COVERAGE_THRESHOLD,
    )
    parser.add_argument(
        "--support-redundancy-beta",
        type=float,
        default=DEFAULT_SUPPORT_REDUNDANCY_BETA,
    )
    parser.add_argument(
        "--support-token-exponent-tau",
        type=float,
        default=DEFAULT_SUPPORT_TOKEN_EXPONENT_TAU,
    )
    parser.add_argument(
        "--support-min-unit-score", type=float, default=DEFAULT_SUPPORT_MIN_UNIT_SCORE
    )
    parser.add_argument(
        "--support-max-selected-units",
        type=int,
        default=DEFAULT_SUPPORT_MAX_SELECTED_UNITS,
    )
    parser.add_argument(
        "--support-node-induction-max-tokens",
        type=int,
        default=DEFAULT_SUPPORT_NODE_INDUCTION_MAX_TOKENS,
    )
    parser.add_argument(
        "--support-teacher-max-tokens",
        type=int,
        default=DEFAULT_SUPPORT_TEACHER_MAX_TOKENS,
    )
    parser.add_argument(
        "--support-prompt-order",
        choices=["anchor_then_node_doc_order", "node_doc_order", "document_order"],
        default=DEFAULT_SUPPORT_PROMPT_ORDER,
    )
    parser.add_argument(
        "--support-score-cache-path", default=DEFAULT_SUPPORT_SCORE_CACHE_PATH
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
        crag_ndocs=args.crag_ndocs,
        crag_upper_threshold=args.crag_upper_threshold,
        crag_lower_threshold=args.crag_lower_threshold,
        crag_decompose_mode=args.crag_decompose_mode,
        crag_external_top_k=args.crag_external_top_k,
        crag_evaluator_device=args.crag_evaluator_device,
        macrag_artifacts_dir=args.macrag_artifacts_dir,
        macrag_top_k1=args.macrag_top_k1,
        macrag_top_k2=args.macrag_top_k2,
        macrag_chunk_ext=args.macrag_chunk_ext,
        macrag_merge_version=args.macrag_merge_version,
        recomp_extractive_model=args.recomp_extractive_model,
        recomp_abstractive_model=args.recomp_abstractive_model,
        recomp_abstractive_mode=args.recomp_abstractive_mode,
        recomp_input_docs=args.recomp_input_docs,
        recomp_top_sentences=args.recomp_top_sentences,
        support_candidate_chunks=args.support_candidate_chunks,
        support_node_candidate_catalog_k=args.support_node_candidate_catalog_k,
        support_max_nodes=args.support_max_nodes,
        support_teacher_units_per_node=args.support_teacher_units_per_node,
        support_unit_types=args.support_unit_types,
        support_include_neighbors=args.support_include_neighbors,
        support_neighbor_hops=args.support_neighbor_hops,
        support_max_units_total=args.support_max_units_total,
        support_max_unit_tokens=args.support_max_unit_tokens,
        support_node_coverage_threshold=args.support_node_coverage_threshold,
        support_redundancy_beta=args.support_redundancy_beta,
        support_token_exponent_tau=args.support_token_exponent_tau,
        support_min_unit_score=args.support_min_unit_score,
        support_max_selected_units=args.support_max_selected_units,
        support_node_induction_max_tokens=args.support_node_induction_max_tokens,
        support_teacher_max_tokens=args.support_teacher_max_tokens,
        support_prompt_order=args.support_prompt_order,
        support_score_cache_path=args.support_score_cache_path,
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
