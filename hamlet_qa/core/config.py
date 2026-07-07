"""Configuration defaults for the Hamlet QA harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_READER_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-Reranker-8B"
DEFAULT_TEMPERATURE = 0.0

DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_TOKENIZER_MODEL = DEFAULT_READER_MODEL

DEFAULT_CONTEXT_BUDGETS = [1000]
BASELINE_TREATMENTS = [
    "closed_book",
    "gold_evidence",
    "dense_reranked",
    "sparse_bm25",
]
ORDERING_TREATMENTS = [
    "dense_document_order",
    "dense_random_order",
]
NEW_CONTEXT_ASSEMBLY_TREATMENTS = [
    "setr",
    "domain",
]
METHOD_TREATMENTS = [
    "crag",
    "macrag",
    "recomp_extractive",
    "recomp_abstractive",
]
OUR_METHOD_TREATMENTS = [
    "reader_support",
]
DEFAULT_TREATMENTS = (
    BASELINE_TREATMENTS
    + ORDERING_TREATMENTS
    + NEW_CONTEXT_ASSEMBLY_TREATMENTS
    + METHOD_TREATMENTS
    + OUR_METHOD_TREATMENTS
)
DEFAULT_TOP_K = 50
DEFAULT_SETR_MAX_PASSAGES = DEFAULT_TOP_K
DEFAULT_RANDOM_SEED = 13
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75
DEFAULT_GPU_LAYOUT = "single"
DEFAULT_DOMAIN_KG_PATH = "data/hamlet_domain_kg.yaml"
DEFAULT_CONTEXT_ASSEMBLY_CACHE_DIR = "data/cache"
DEFAULT_SETR_SELECTOR_MAX_TOKENS = 4096

# CRAG: action thresholds over Qwen reranker logits, calibrated by
# `python -m hamlet_qa.cli.calibrate_crag` on runs/qwen_hamlet_probe:
# upper = smallest score with >=0.9 gold precision, lower = non-gold p90.
DEFAULT_CRAG_NDOCS = 10
DEFAULT_CRAG_UPPER_THRESHOLD = 2.5
DEFAULT_CRAG_LOWER_THRESHOLD = 0.875
DEFAULT_CRAG_DECOMPOSE_MODE = "excerption"
DEFAULT_CRAG_EXTERNAL_TOP_K = 5

# MacRAG offline index + query-time scale-up settings (slice sizes follow the
# official 450/300-char summary slicing).
DEFAULT_MACRAG_ARTIFACTS_DIR = "data/macrag"
DEFAULT_MACRAG_TOP_K1 = 100
DEFAULT_MACRAG_TOP_K2 = 7
DEFAULT_MACRAG_CHUNK_EXT = 1
DEFAULT_MACRAG_MERGE_VERSION = 1
DEFAULT_MACRAG_SLICE_SIZE = 450
DEFAULT_MACRAG_SLICE_OVERLAP = 300

# RECOMP compressors (official trained checkpoints; HotpotQA variants).
DEFAULT_RECOMP_EXTRACTIVE_MODEL = "fangyuan/hotpotqa_extractive_compressor"
DEFAULT_RECOMP_ABSTRACTIVE_MODEL = "fangyuan/hotpotqa_abstractive"
DEFAULT_RECOMP_ABSTRACTIVE_MODE = "t5"
DEFAULT_RECOMP_INPUT_DOCS = 5
DEFAULT_RECOMP_TOP_SENTENCES = 5

# reader_support: our Reader-Supervised Evidence Support Assembler. The reader
# model judges how well each source unit supports each evidence need; a budgeted
# greedy coverage objective then selects a compact, source-faithful context.
DEFAULT_SUPPORT_CANDIDATE_CHUNKS = 30
DEFAULT_SUPPORT_NODE_CANDIDATE_CATALOG_K = 20
DEFAULT_SUPPORT_MAX_NODES = 5
DEFAULT_SUPPORT_TEACHER_UNITS_PER_NODE = 12
DEFAULT_SUPPORT_UNIT_TYPES = "chunk,sentence,line_span,neighbor_left,neighbor_right"
DEFAULT_SUPPORT_INCLUDE_NEIGHBORS = True
DEFAULT_SUPPORT_NEIGHBOR_HOPS = 1
DEFAULT_SUPPORT_MAX_UNITS_TOTAL = 200
DEFAULT_SUPPORT_MAX_UNIT_TOKENS = 512
DEFAULT_SUPPORT_NODE_COVERAGE_THRESHOLD = 0.85
DEFAULT_SUPPORT_REDUNDANCY_BETA = 0.15
DEFAULT_SUPPORT_TOKEN_EXPONENT_TAU = 0.7
DEFAULT_SUPPORT_MIN_UNIT_SCORE = 0.45
DEFAULT_SUPPORT_MAX_SELECTED_UNITS = 8
DEFAULT_SUPPORT_NODE_INDUCTION_MAX_TOKENS = 1024
DEFAULT_SUPPORT_TEACHER_MAX_TOKENS = 384
DEFAULT_SUPPORT_PROMPT_ORDER = "anchor_then_node_doc_order"
DEFAULT_SUPPORT_SCORE_CACHE_PATH = "data/cache/reader_support_cache.json"

# evidence-planning experiment (plan_fixed / plan_dynamic). Opt-in treatments
# that isolate "how the LLM breaks down + plans the retrieval procedure". Prompt
# variants are selected by name (see hamlet_qa/features/evidence_plan/prompts.py).
DEFAULT_PLAN_DECOMP_PROMPT = "list_requirements"
DEFAULT_PLAN_PLANNER_PROMPT = "contract_v1"
DEFAULT_PLAN_FOLLOWUP_PROMPT = "rewrite_with_evidence"
DEFAULT_PLAN_RETRIEVAL_MODE = "parallel"
DEFAULT_PLAN_SUPPORT_POLICY = "reranker"
DEFAULT_PLAN_SELECTION_POLICY = "greedy_coverage"
DEFAULT_PLAN_ORDERING_POLICY = "document_order"
DEFAULT_PLAN_NODE_TOP_K = 10
DEFAULT_PLAN_MAX_NODES = 5
DEFAULT_PLAN_CATALOG_K = 20
DEFAULT_PLAN_MIN_SUPPORT = 0.5
DEFAULT_PLAN_SUPPORT_TEMP = 1.0
DEFAULT_PLAN_COVERAGE_THRESHOLD = 0.85
DEFAULT_PLAN_REDUNDANCY_BETA = 0.15
DEFAULT_PLAN_TOKEN_EXPONENT_TAU = 0.7
DEFAULT_PLAN_MAX_SELECTED_UNITS = 8
DEFAULT_PLAN_LLM_MAX_TOKENS = 1024
DEFAULT_PLAN_FOLLOWUP_MAX_TOKENS = 256
DEFAULT_PLAN_TEACHER_MAX_TOKENS = 384
DEFAULT_PLAN_CACHE_PATH = "data/cache/evidence_plan_cache.json"

GPU_LAYOUTS = {
    "single": {
        "embedding_device": "cuda",
        "reranker_device": "cuda",
        "reader_device": "cuda",
    },
    "a40-2gpu": {
        "embedding_device": "cuda:0",
        "reranker_device": "cuda:0",
        "reader_device": "cuda:1",
    },
    "a40-3gpu": {
        "embedding_device": "cuda:0",
        "reranker_device": "cuda:1",
        "reader_device": "cuda:2",
    },
}

REASONING_SKILLS = [
    "local_fact",
    "speaker_attribution",
    "scene_local_context",
    "cross_scene_bridge",
    "temporal_order",
    "entity_state_tracking",
    "deception_or_mistaken_belief",
    "distractor_contrast",
    "unanswerable",
    "theme_or_symbolism",
]
QUESTION_CATEGORIES = REASONING_SKILLS


@dataclass
class RunConfig:
    """Serializable settings for one Hamlet QA run."""

    document_path: str = "data/hamlet.txt"
    chunks_path: str = "data/hamlet_chunks.jsonl"
    questions_path: str = "data/hamlet_questions.json"
    output_dir: str = "runs"
    run_name: str = "hamlet_probe"

    tokenizer_model: str = DEFAULT_TOKENIZER_MODEL
    reader_model: str = DEFAULT_READER_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reranker_model: str | None = DEFAULT_RERANKER_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_new_tokens: int = 512

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    context_budgets: list[int] = field(
        default_factory=lambda: DEFAULT_CONTEXT_BUDGETS.copy()
    )
    treatments: list[str] = field(default_factory=lambda: DEFAULT_TREATMENTS.copy())
    top_k: int = DEFAULT_TOP_K
    random_seed: int = DEFAULT_RANDOM_SEED
    gpu_layout: str = DEFAULT_GPU_LAYOUT

    embedding_batch_size: int = 64
    embedding_device: str = "cuda"
    reranker_batch_size: int = 8
    reranker_device: str = "cuda"
    reader_device: str = "cuda"
    bm25_k1: float = DEFAULT_BM25_K1
    bm25_b: float = DEFAULT_BM25_B
    domain_kg_path: str = DEFAULT_DOMAIN_KG_PATH
    context_assembly_cache_dir: str = DEFAULT_CONTEXT_ASSEMBLY_CACHE_DIR
    setr_max_passages: int = DEFAULT_SETR_MAX_PASSAGES
    setr_selector_max_tokens: int = DEFAULT_SETR_SELECTOR_MAX_TOKENS
    crag_ndocs: int = DEFAULT_CRAG_NDOCS
    crag_upper_threshold: float = DEFAULT_CRAG_UPPER_THRESHOLD
    crag_lower_threshold: float = DEFAULT_CRAG_LOWER_THRESHOLD
    crag_decompose_mode: str = DEFAULT_CRAG_DECOMPOSE_MODE
    crag_external_top_k: int = DEFAULT_CRAG_EXTERNAL_TOP_K
    crag_evaluator_device: str | None = None
    macrag_artifacts_dir: str = DEFAULT_MACRAG_ARTIFACTS_DIR
    macrag_top_k1: int = DEFAULT_MACRAG_TOP_K1
    macrag_top_k2: int = DEFAULT_MACRAG_TOP_K2
    macrag_chunk_ext: int = DEFAULT_MACRAG_CHUNK_EXT
    macrag_merge_version: int = DEFAULT_MACRAG_MERGE_VERSION
    recomp_extractive_model: str = DEFAULT_RECOMP_EXTRACTIVE_MODEL
    recomp_abstractive_model: str = DEFAULT_RECOMP_ABSTRACTIVE_MODEL
    recomp_abstractive_mode: str = DEFAULT_RECOMP_ABSTRACTIVE_MODE
    recomp_input_docs: int = DEFAULT_RECOMP_INPUT_DOCS
    recomp_top_sentences: int = DEFAULT_RECOMP_TOP_SENTENCES
    support_candidate_chunks: int = DEFAULT_SUPPORT_CANDIDATE_CHUNKS
    support_node_candidate_catalog_k: int = DEFAULT_SUPPORT_NODE_CANDIDATE_CATALOG_K
    support_max_nodes: int = DEFAULT_SUPPORT_MAX_NODES
    support_teacher_units_per_node: int = DEFAULT_SUPPORT_TEACHER_UNITS_PER_NODE
    support_unit_types: str = DEFAULT_SUPPORT_UNIT_TYPES
    support_include_neighbors: bool = DEFAULT_SUPPORT_INCLUDE_NEIGHBORS
    support_neighbor_hops: int = DEFAULT_SUPPORT_NEIGHBOR_HOPS
    support_max_units_total: int = DEFAULT_SUPPORT_MAX_UNITS_TOTAL
    support_max_unit_tokens: int = DEFAULT_SUPPORT_MAX_UNIT_TOKENS
    support_node_coverage_threshold: float = DEFAULT_SUPPORT_NODE_COVERAGE_THRESHOLD
    support_redundancy_beta: float = DEFAULT_SUPPORT_REDUNDANCY_BETA
    support_token_exponent_tau: float = DEFAULT_SUPPORT_TOKEN_EXPONENT_TAU
    support_min_unit_score: float = DEFAULT_SUPPORT_MIN_UNIT_SCORE
    support_max_selected_units: int = DEFAULT_SUPPORT_MAX_SELECTED_UNITS
    support_node_induction_max_tokens: int = DEFAULT_SUPPORT_NODE_INDUCTION_MAX_TOKENS
    support_teacher_max_tokens: int = DEFAULT_SUPPORT_TEACHER_MAX_TOKENS
    support_prompt_order: str = DEFAULT_SUPPORT_PROMPT_ORDER
    support_score_cache_path: str = DEFAULT_SUPPORT_SCORE_CACHE_PATH
    plan_decomp_prompt: str = DEFAULT_PLAN_DECOMP_PROMPT
    plan_planner_prompt: str = DEFAULT_PLAN_PLANNER_PROMPT
    plan_followup_prompt: str = DEFAULT_PLAN_FOLLOWUP_PROMPT
    plan_retrieval_mode: str = DEFAULT_PLAN_RETRIEVAL_MODE
    plan_support_policy: str = DEFAULT_PLAN_SUPPORT_POLICY
    plan_selection_policy: str = DEFAULT_PLAN_SELECTION_POLICY
    plan_ordering_policy: str = DEFAULT_PLAN_ORDERING_POLICY
    plan_node_top_k: int = DEFAULT_PLAN_NODE_TOP_K
    plan_max_nodes: int = DEFAULT_PLAN_MAX_NODES
    plan_catalog_k: int = DEFAULT_PLAN_CATALOG_K
    plan_min_support: float = DEFAULT_PLAN_MIN_SUPPORT
    plan_support_temp: float = DEFAULT_PLAN_SUPPORT_TEMP
    plan_coverage_threshold: float = DEFAULT_PLAN_COVERAGE_THRESHOLD
    plan_redundancy_beta: float = DEFAULT_PLAN_REDUNDANCY_BETA
    plan_token_exponent_tau: float = DEFAULT_PLAN_TOKEN_EXPONENT_TAU
    plan_max_selected_units: int = DEFAULT_PLAN_MAX_SELECTED_UNITS
    plan_llm_max_tokens: int = DEFAULT_PLAN_LLM_MAX_TOKENS
    plan_followup_max_tokens: int = DEFAULT_PLAN_FOLLOWUP_MAX_TOKENS
    plan_teacher_max_tokens: int = DEFAULT_PLAN_TEACHER_MAX_TOKENS
    plan_cache_path: str = DEFAULT_PLAN_CACHE_PATH
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    prepare_only: bool = False
    overwrite: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name
