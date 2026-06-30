"""Shared executor for both planning treatments.

`plan_fixed` and `plan_dynamic` call exactly this code; they differ only in
where the policies come from (fixed flags vs. a parsed contract). Steps:
decompose-result nodes -> (parallel | sequential) per-node retrieval -> support
scoring (reranker | teacher) -> selection (greedy_coverage | top_per_node) ->
ordering. Selected context is whole source chunks (real chunk ids).
"""

from __future__ import annotations

from typing import Any, Callable

from hamlet_qa.features.evidence_plan.retrieve import (
    evidence_snippets,
    normalized_reranker_support,
    retrieve_for_node,
)
from hamlet_qa.core.evidence.coverage import greedy_select
from hamlet_qa.core.evidence.schema import EvidenceNode, EvidenceUnit


def _retrieval_order(nodes: list[EvidenceNode]) -> list[EvidenceNode]:
    return sorted(nodes, key=lambda node: (node.order_index, node.node_id))


def _chunk_unit(chunk: dict[str, Any]) -> EvidenceUnit:
    chunk_id = str(chunk["chunk_id"])
    return EvidenceUnit(
        unit_id=chunk_id,
        unit_type="chunk",
        text=str(chunk["text"]),
        source_chunk_ids=[chunk_id],
        primary_chunk_id=chunk_id,
        token_count=int(chunk["token_count"]),
        source_order_key=[int(chunk["global_index"]), 0],
        act=chunk.get("act"),
        scene=chunk.get("scene"),
        scene_title=chunk.get("scene_title"),
        scene_id=chunk.get("scene_id"),
        global_index_start=int(chunk["global_index"]),
        global_index_end=int(chunk["global_index"]),
    )


def execute_plan(
    question_text: str,
    nodes: list[EvidenceNode],
    *,
    node_retriever: Any,
    chunk_lookup: dict[str, dict[str, Any]],
    context_budget: int,
    retrieval_mode: str,
    support_policy: str,
    selection_policy: str,
    ordering_policy: str,
    node_top_k: int,
    support_temp: float,
    coverage_threshold: float,
    redundancy_beta: float,
    token_exponent_tau: float,
    min_support: float,
    max_selected_units: int,
    teacher_scorer: Any = None,
    reformulate: Callable[[EvidenceNode, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    trace: dict[str, Any] = {"retrieval_mode": retrieval_mode, "support_policy": support_policy}

    # --- Retrieval (parallel or sequential with follow-up reformulation) ------
    node_rows: dict[str, list[dict[str, Any]]] = {}
    retrieval_trace: list[dict[str, Any]] = []
    gathered: list[str] = []
    for node in _retrieval_order(nodes):
        query = node.node_query
        reformulated = None
        if retrieval_mode == "sequential" and node.depends_on and reformulate is not None:
            result = reformulate(node, evidence_snippets(gathered, chunk_lookup))
            query = result.get("query") or node.node_query
            reformulated = {"query": query, "cache_hit": result.get("cache_hit")}
        rows = [
            row
            for row in retrieve_for_node(node_retriever, query, node_top_k)
            if str(row["chunk_id"]) in chunk_lookup
        ]
        node_rows[node.node_id] = rows
        if retrieval_mode == "sequential":
            gathered.extend(str(row["chunk_id"]) for row in rows[: max(1, node_top_k // 2)])
        retrieval_trace.append(
            {
                "node_id": node.node_id,
                "query_used": query,
                "reformulated": reformulated,
                "retrieved": [
                    {"chunk_id": str(row["chunk_id"]), "raw_score": row.get("rerank_score", row.get("score"))}
                    for row in rows
                ],
            }
        )
    trace["per_node_retrieval"] = retrieval_trace

    # --- Units (one per retrieved chunk) + support matrix ---------------------
    unit_ids: list[str] = []
    units: list[EvidenceUnit] = []
    seen: set[str] = set()
    for node in nodes:
        for row in node_rows.get(node.node_id, []):
            chunk_id = str(row["chunk_id"])
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            unit_ids.append(chunk_id)
            units.append(_chunk_unit(chunk_lookup[chunk_id]))
    units_by_id = {unit.unit_id: unit for unit in units}

    matrix: dict[tuple[str, str], float] = {}
    support_detail: dict[str, list[dict[str, Any]]] = {}
    teacher_calls = 0
    for node in nodes:
        rows = node_rows.get(node.node_id, [])
        node_detail: list[dict[str, Any]] = []
        for row in rows:
            chunk_id = str(row["chunk_id"])
            if support_policy == "teacher" and teacher_scorer is not None:
                support = teacher_scorer.score(question_text, node, units_by_id[chunk_id]).support_score
                teacher_calls += 1
            else:
                support = normalized_reranker_support(row, support_temp)
            matrix[(node.node_id, chunk_id)] = support
            node_detail.append({"chunk_id": chunk_id, "support": round(support, 4)})
        node_detail.sort(key=lambda item: item["support"], reverse=True)
        support_detail[node.node_id] = node_detail
    trace["support_scoring"] = {"policy": support_policy, "teacher_calls": teacher_calls, "per_node": support_detail}

    # --- Selection ------------------------------------------------------------
    if selection_policy == "top_per_node":
        selected, selection_trace = _select_top_per_node(
            units_by_id, nodes, matrix, context_budget, min_support, max_selected_units
        )
    else:
        selection_policy = "greedy_coverage"
        greedy = greedy_select(
            units,
            nodes,
            matrix,
            context_budget=context_budget,
            beta=redundancy_beta,
            tau=token_exponent_tau,
            min_unit_score=min_support,
            coverage_threshold=coverage_threshold,
            max_selected=max_selected_units,
        )
        selected = greedy["selected"]
        selection_trace = {
            "selection_steps": greedy["steps"],
            "coverage_progress": greedy["progress"],
            "final_coverage": {n.node_id: round(1.0 - greedy["miss"][n.node_id], 4) for n in nodes},
            "num_selectable": len(greedy["selectable"]),
        }
    trace["selection_policy"] = selection_policy
    trace["selection"] = selection_trace

    if not selected:
        trace["empty_reason"] = "no candidate cleared min_support"
        return {"selected_chunk_ids": [], "selected_chunks": [], "trace": trace, "empty": True}

    # --- Ordering -------------------------------------------------------------
    ordered = _order_units(selected, nodes, matrix, ordering_policy)
    selected_chunks = [dict(chunk_lookup[unit.unit_id]) for unit in ordered]
    selected_ids = [str(chunk["chunk_id"]) for chunk in selected_chunks]
    trace["final_ordering"] = selected_ids
    trace["final_token_count"] = sum(int(chunk["token_count"]) for chunk in selected_chunks)
    return {"selected_chunk_ids": selected_ids, "selected_chunks": selected_chunks, "trace": trace, "empty": False}


def _select_top_per_node(
    units_by_id: dict[str, EvidenceUnit],
    nodes: list[EvidenceNode],
    matrix: dict[tuple[str, str], float],
    context_budget: int,
    min_support: float,
    max_selected: int,
) -> tuple[list[EvidenceUnit], dict[str, Any]]:
    """Keep each node's best units, then budget-fill in support order."""
    import math as _math

    per_node = max(1, _math.ceil(max_selected / max(1, len(nodes))))
    ranked: list[tuple[float, str]] = []
    chosen_ids: list[str] = []
    for node in nodes:
        scored = [
            (matrix.get((node.node_id, uid), 0.0), uid)
            for uid in units_by_id
            if matrix.get((node.node_id, uid), 0.0) >= min_support
        ]
        scored.sort(reverse=True)
        for support, uid in scored[:per_node]:
            ranked.append((support, uid))
            if uid not in chosen_ids:
                chosen_ids.append(uid)
    ranked.sort(reverse=True)
    selected: list[EvidenceUnit] = []
    total = 0
    seen: set[str] = set()
    for support, uid in ranked:
        if uid in seen:
            continue
        unit = units_by_id[uid]
        if total + unit.token_count > context_budget or len(selected) >= max_selected:
            continue
        seen.add(uid)
        selected.append(unit)
        total += unit.token_count
    return selected, {"per_node_keep": per_node, "selected_unit_ids": [u.unit_id for u in selected]}


def _strongest_node(unit: EvidenceUnit, nodes: list[EvidenceNode], matrix: dict[tuple[str, str], float]) -> EvidenceNode:
    return max(nodes, key=lambda n: matrix.get((n.node_id, unit.unit_id), 0.0))


def _order_units(
    selected: list[EvidenceUnit],
    nodes: list[EvidenceNode],
    matrix: dict[tuple[str, str], float],
    policy: str,
) -> list[EvidenceUnit]:
    def doc_key(unit: EvidenceUnit) -> tuple:
        return (tuple(unit.source_order_key), unit.unit_id)

    def node_doc_key(unit: EvidenceUnit) -> tuple:
        return (_strongest_node(unit, nodes, matrix).order_index, tuple(unit.source_order_key), unit.unit_id)

    if policy == "document_order":
        return sorted(selected, key=doc_key)
    if policy == "node_order":
        return sorted(selected, key=node_doc_key)
    # anchor_first
    anchor = max(
        selected,
        key=lambda u: max((matrix.get((n.node_id, u.unit_id), 0.0) for n in nodes), default=0.0),
    )
    rest = sorted((u for u in selected if u is not anchor), key=node_doc_key)
    return [anchor, *rest]
