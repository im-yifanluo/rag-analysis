"""Shared budgeted submodular coverage selection.

`greedy_select` maximizes node coverage (noisy-OR over support scores) minus a
redundancy penalty, token-normalized, under a budget. Pure and side-effect-free
so it is directly testable. Also exposes `lexical_prior`, a cheap query-term
overlap used by callers for prefiltering. No feature-package imports.
"""

from __future__ import annotations

import math
import re
from typing import Any

from hamlet_qa.core.evidence.schema import EvidenceNode, EvidenceUnit

_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "was", "were",
    "what", "who", "whom", "how", "why", "when", "where", "which", "does", "do",
    "did", "his", "her", "their", "they", "he", "she", "it", "that", "this",
    "for", "on", "at", "by", "with", "as", "from", "into", "be", "been",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _tokens(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z0-9']+", _normalize(text))
        if word not in _STOPWORDS and len(word) > 1
    }


def lexical_prior(node: EvidenceNode, unit: EvidenceUnit) -> float:
    """Cheap query-term coverage prior used ONLY to prefilter scoring."""
    query_tokens = _tokens(f"{node.need} {node.node_query}")
    if not query_tokens:
        return 0.0
    unit_tokens = _tokens(unit.text)
    return len(query_tokens & unit_tokens) / len(query_tokens)


def _text_jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _cosine(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    na = math.sqrt(sum(x * x for x in vec_a))
    nb = math.sqrt(sum(y * y for y in vec_b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _support_vector(
    unit_id: str, nodes: list[EvidenceNode], matrix: dict[tuple[str, str], float]
) -> list[float]:
    return [matrix.get((node.node_id, unit_id), 0.0) for node in nodes]


def _redundancy(
    unit: EvidenceUnit,
    selected: list[EvidenceUnit],
    nodes: list[EvidenceNode],
    matrix: dict[tuple[str, str], float],
) -> float:
    if not selected:
        return 0.0
    unit_vec = _support_vector(unit.unit_id, nodes, matrix)
    unit_sources = set(unit.source_chunk_ids)
    best = 0.0
    for other in selected:
        text_overlap = _text_jaccard(unit.text, other.text)
        support_cos = _cosine(unit_vec, _support_vector(other.unit_id, nodes, matrix))
        source_bonus = 0.5 if unit_sources & set(other.source_chunk_ids) else 0.0
        red = 0.6 * text_overlap + 0.4 * support_cos + source_bonus
        best = max(best, red)
    return max(0.0, min(1.0, best))


def greedy_select(
    units: list[EvidenceUnit],
    nodes: list[EvidenceNode],
    matrix: dict[tuple[str, str], float],
    *,
    context_budget: int,
    beta: float,
    tau: float,
    min_unit_score: float,
    coverage_threshold: float,
    max_selected: int,
) -> dict[str, Any]:
    """Budgeted greedy maximization of node coverage minus redundancy.

    node coverage_j(S) = 1 - prod_{u in S}(1 - support[j][u]); the per-step pick
    maximizes (coverage gain - beta*redundancy) / token_count^tau. Pure and
    side-effect-free so it is directly testable.
    """

    def unit_score(node_id: str, unit_id: str) -> float:
        return matrix.get((node_id, unit_id), 0.0)

    selectable = [
        unit
        for unit in units
        if max((unit_score(n.node_id, unit.unit_id) for n in nodes), default=0.0)
        >= min_unit_score
    ]
    miss = {node.node_id: 1.0 for node in nodes}
    selected: list[EvidenceUnit] = []
    remaining_budget = context_budget
    steps: list[dict[str, Any]] = []
    progress: list[dict[str, float]] = [{node.node_id: 0.0 for node in nodes}]

    while len(selected) < max_selected:
        if all((1.0 - miss[n.node_id]) >= coverage_threshold for n in nodes):
            break
        best_unit: EvidenceUnit | None = None
        best_gain = 0.0
        best_detail: dict[str, Any] = {}
        for unit in selectable:
            if unit in selected or unit.token_count > remaining_budget:
                continue
            cov_gain = sum(
                miss[n.node_id] * unit_score(n.node_id, unit.unit_id) for n in nodes
            )
            redundancy = _redundancy(unit, selected, nodes, matrix)
            marginal = cov_gain - beta * redundancy
            gain_per_token = marginal / (max(1, unit.token_count) ** tau)
            if gain_per_token > best_gain:
                best_gain = gain_per_token
                best_unit = unit
                best_detail = {
                    "marginal_gain": round(marginal, 5),
                    "coverage_gain": round(cov_gain, 5),
                    "redundancy_penalty": round(redundancy, 5),
                    "gain_per_token": round(gain_per_token, 6),
                }
        if best_unit is None or best_gain <= 0.0:
            break
        selected.append(best_unit)
        remaining_budget -= best_unit.token_count
        for node in nodes:
            miss[node.node_id] *= 1.0 - unit_score(node.node_id, best_unit.unit_id)
        steps.append(
            {
                "unit_id": best_unit.unit_id,
                "unit_type": best_unit.unit_type,
                "token_count": best_unit.token_count,
                "selected_support_scores": {
                    n.node_id: unit_score(n.node_id, best_unit.unit_id) for n in nodes
                },
                **best_detail,
            }
        )
        progress.append({n.node_id: 1.0 - miss[n.node_id] for n in nodes})

    return {
        "selected": selected,
        "steps": steps,
        "progress": progress,
        "miss": miss,
        "remaining_budget": remaining_budget,
        "selectable": selectable,
    }
