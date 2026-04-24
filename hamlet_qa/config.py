"""Configuration defaults for the Hamlet QA harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_READER_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-m-v1.5"
DEFAULT_TEMPERATURE = 0.0

DEFAULT_CHUNK_SIZE = 256
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_TOKENIZER_MODEL = "Qwen/Qwen2.5-7B-Instruct"

DEFAULT_CONTEXT_BUDGETS = [1000]
DEFAULT_TREATMENTS = [
    "closed_book",
    "gold_evidence",
    "gold_evidence_neighbors",
    "dense_relevance",
    "dense_relevance_neighbors",
]
DEFAULT_TOP_K = 50
DEFAULT_NEIGHBOR_WINDOW = 1

REASONING_SKILLS = [
    "local_fact",
    "speaker_attribution",
    "scene_local_context",
    "cross_scene_bridge",
    "temporal_order",
    "entity_state_tracking",
    "deception_or_mistaken_belief",
    "causal_explanation",
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
    temperature: float = DEFAULT_TEMPERATURE
    max_new_tokens: int = 512

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    context_budgets: list[int] = field(
        default_factory=lambda: DEFAULT_CONTEXT_BUDGETS.copy()
    )
    treatments: list[str] = field(default_factory=lambda: DEFAULT_TREATMENTS.copy())
    top_k: int = DEFAULT_TOP_K
    neighbor_window: int = DEFAULT_NEIGHBOR_WINDOW

    embedding_batch_size: int = 64
    embedding_device: str = "cuda"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    prepare_only: bool = False
    overwrite: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name
