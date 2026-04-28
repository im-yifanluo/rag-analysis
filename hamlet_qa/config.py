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
DEFAULT_TREATMENTS = [
    "closed_book",
    "gold_evidence",
    "dense_reranked",
    "dense_document_order",
    "dense_random_order",
    "sparse_bm25",
]
DEFAULT_TOP_K = 50
DEFAULT_RANDOM_SEED = 13
DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75

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

    document_path: str = "hamlet.txt"
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

    embedding_batch_size: int = 64
    embedding_device: str = "cuda"
    reranker_batch_size: int = 8
    reranker_device: str = "cuda"
    bm25_k1: float = DEFAULT_BM25_K1
    bm25_b: float = DEFAULT_BM25_B
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    prepare_only: bool = False
    overwrite: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name
