"""Shared context assembly contract and chunk-selection utilities."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from hamlet_qa.core.questions import Question


RetrievalSource = Literal["none", "dense", "sparse", "macrag"]


@dataclass(frozen=True)
class ContextAssemblyRequest:
    """Everything a context assembly strategy needs for one question."""

    question: Question
    treatment: str
    context_budget: int
    chunk_lookup: dict[str, dict[str, Any]]
    doc_order_ids: list[str]
    retrieval_trace: list[dict[str, Any]] = field(default_factory=list)
    random_seed: int = 13
    domain_kg: Any = None
    selector_model: Any = None
    setr_cache_path: Path | None = None
    setr_max_passages: int = 50
    setr_selector_max_tokens: int = 4096
    feature_params: dict[str, Any] = field(default_factory=dict)
    feature_handles: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextAssemblyResult:
    """Prompt-ready context plus trace metadata."""

    selected_chunk_ids: list[str]
    selected_chunks: list[dict[str, Any]]
    original_hit_chunk_ids: list[str] = field(default_factory=list)
    retrieval_trace: list[dict[str, Any]] = field(default_factory=list)
    retrieval_method: str = "none"
    prompt_order: str = "none"
    context_assembly_trace: dict[str, Any] | None = None

    @property
    def context_tokens(self) -> int:
        return context_token_count(self.selected_chunks)


AssemblyFn = Callable[[ContextAssemblyRequest], ContextAssemblyResult]


@dataclass(frozen=True)
class TreatmentSpec:
    """Registry metadata for one experiment treatment."""

    name: str
    assemble: AssemblyFn
    retrieval_source: RetrievalSource = "none"
    uses_domain_kg: bool = False
    uses_llm_assembly: bool = False


def chunks_by_id(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {chunk["chunk_id"]: dict(chunk) for chunk in chunks}


def document_order_chunk_ids(chunks: list[dict[str, Any]]) -> list[str]:
    return [
        chunk["chunk_id"]
        for chunk in sorted(chunks, key=lambda item: int(item["global_index"]))
    ]


def dedupe_preserve_order(chunk_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for chunk_id in chunk_ids:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        deduped.append(chunk_id)
    return deduped


def dedupe_existing_chunk_ids(
    candidate_chunk_ids: list[str],
    chunk_lookup: dict[str, dict[str, Any]],
) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for chunk_id in candidate_chunk_ids:
        if chunk_id in seen or chunk_id not in chunk_lookup:
            continue
        seen.add(chunk_id)
        deduped.append(chunk_id)
    return deduped


def candidate_rank_map(candidate_chunk_ids: list[str]) -> dict[str, int]:
    return {chunk_id: index for index, chunk_id in enumerate(candidate_chunk_ids)}


def sort_by_document_order(
    chunk_ids: list[str],
    chunk_lookup: dict[str, dict[str, Any]],
) -> list[str]:
    return sorted(
        dedupe_preserve_order(chunk_ids),
        key=lambda chunk_id: int(chunk_lookup[chunk_id]["global_index"]),
    )


def stable_random_order(chunk_ids: list[str], random_seed: int, salt: str) -> list[str]:
    ordered = list(chunk_ids)
    seed_material = f"{random_seed}:{salt}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    random.Random(seed).shuffle(ordered)
    return ordered


def select_chunk_ids_for_budget(
    candidate_chunk_ids: list[str],
    chunk_lookup: dict[str, dict[str, Any]],
    context_budget: int,
) -> list[str]:
    selected: list[str] = []
    total_tokens = 0
    for chunk_id in dedupe_preserve_order(candidate_chunk_ids):
        chunk = chunk_lookup[chunk_id]
        token_count = int(chunk["token_count"])
        if total_tokens + token_count > context_budget:
            continue
        selected.append(chunk_id)
        total_tokens += token_count
    return selected


def selected_chunks(
    selected_chunk_ids: list[str],
    chunk_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [dict(chunk_lookup[chunk_id]) for chunk_id in selected_chunk_ids]


def context_token_count(chunks: list[dict[str, Any]]) -> int:
    return sum(int(chunk["token_count"]) for chunk in chunks)


def truncate_text_to_word_budget(text: str, word_budget: int) -> str:
    """Trim text to at most word_budget whitespace-delimited words."""
    if word_budget <= 0:
        return ""
    words = text.split()
    if len(words) <= word_budget:
        return text
    return " ".join(words[:word_budget])


def make_pseudo_chunk(
    chunk_id: str,
    text: str,
    scene_title: str,
    scene_id: str = "synthetic",
) -> dict[str, Any]:
    """Build a synthetic context chunk (domain-scaffold convention).

    Token count uses whitespace word count, matching the existing
    `domain_scaffold` pseudo-chunk convention.
    """
    word_count = len(text.split())
    return {
        "chunk_id": chunk_id,
        "global_index": -1,
        "act": 0,
        "scene": 0,
        "scene_id": scene_id,
        "scene_title": scene_title,
        "chunk_in_scene": 0,
        "start_token": 0,
        "end_token": word_count,
        "token_count": word_count,
        "text": text,
    }
