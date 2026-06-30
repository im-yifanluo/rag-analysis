"""Per-node dense retrieval and support-score normalization.

The `node_retriever` handle is a `core.retrieval.DenseRetriever` (embedder +
FAISS + optional reranker); `.retrieve(query, top_k)` returns reranked rows. For
the `reranker` support policy we squash the (unbounded) reranker logit through a
sigmoid so it lands in [0, 1] and the noisy-OR coverage in `greedy_select` is
well defined: sigmoid(0)=0.5, and on the Qwen reranker the CRAG-calibrated
"good" band (~0.9–2.5) maps to ~0.7–0.92, matching the support rubric.
"""

from __future__ import annotations

import math
import re
from typing import Any


class NodeRetrieverLike:
    """Protocol marker: anything with .retrieve(query, top_k) -> list[dict]."""

    def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:  # pragma: no cover
        ...


def sigmoid(value: float, temperature: float = 1.0) -> float:
    temperature = temperature if temperature and temperature > 0 else 1.0
    try:
        return 1.0 / (1.0 + math.exp(-value / temperature))
    except OverflowError:
        return 0.0 if value < 0 else 1.0


def raw_support_score(row: dict[str, Any]) -> float:
    """The raw retrieval signal for a candidate row, reranker logit preferred."""
    for key in ("rerank_score", "score", "dense_score"):
        if row.get(key) is not None:
            return float(row[key])
    return 0.0


def normalized_reranker_support(row: dict[str, Any], temperature: float) -> float:
    return max(0.0, min(1.0, sigmoid(raw_support_score(row), temperature)))


def retrieve_for_node(
    node_retriever: NodeRetrieverLike, query: str, top_k: int
) -> list[dict[str, Any]]:
    rows = node_retriever.retrieve(query, top_k)
    return [dict(row) for row in rows]


def evidence_snippets(
    chunk_ids: list[str],
    chunk_lookup: dict[str, dict[str, Any]],
    max_chunks: int = 3,
    max_chars: int = 300,
) -> str:
    """Compact evidence string fed to the sequential follow-up reformulation."""
    lines: list[str] = []
    for chunk_id in chunk_ids[:max_chunks]:
        chunk = chunk_lookup.get(chunk_id)
        if chunk is None:
            continue
        excerpt = re.sub(r"\s+", " ", str(chunk.get("text", "")).strip())[:max_chars]
        lines.append(f"- [{chunk_id}] {excerpt}")
    return "\n".join(lines) if lines else "(none yet)"
