"""Stage 4: budgeted, support-based context assembly and ordering.

Ties the pipeline together: induce evidence nodes, build source-faithful
candidate units, score (node, unit) support with the reader-teacher, then
greedily select a compact, non-redundant, budget-respecting set that covers the
nodes, and order it coherently. The final context is always source-extractive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hamlet_qa.core.context import ContextAssemblyRequest, ContextAssemblyResult
from hamlet_qa.core.evidence.catalog import build_candidate_catalog
from hamlet_qa.core.evidence.coverage import greedy_select, lexical_prior
from hamlet_qa.core.evidence.schema import EvidenceNode, EvidenceUnit, SupportScore
from hamlet_qa.core.evidence.support_teacher import ReaderTeacherSupportScorer
from hamlet_qa.core.llm_cache import JsonKVCache
from hamlet_qa.features.reader_support.nodes import induce_nodes
from hamlet_qa.features.reader_support.units import build_units

READER_SUPPORT_DEVIATIONS = [
    "Our own prototype, not a reproduction of any single paper.",
    "Support is scored by the reader model as a teacher (no hand-tuned blend of "
    "reranker/BM25/entity scores); a lexical prior is used only to prefilter "
    "which (node, unit) pairs get scored.",
    "Final context is source-extractive only; no abstractive pseudo-evidence.",
]

_DEFAULTS: dict[str, Any] = {
    "support_candidate_chunks": 30,
    "support_node_candidate_catalog_k": 20,
    "support_max_nodes": 5,
    "support_teacher_units_per_node": 12,
    "support_unit_types": "chunk,sentence,line_span,neighbor_left,neighbor_right",
    "support_include_neighbors": True,
    "support_neighbor_hops": 1,
    "support_max_units_total": 200,
    "support_max_unit_tokens": 512,
    "support_node_coverage_threshold": 0.85,
    "support_redundancy_beta": 0.15,
    "support_token_exponent_tau": 0.7,
    "support_min_unit_score": 0.45,
    "support_max_selected_units": 8,
    "support_node_induction_max_tokens": 1024,
    "support_teacher_max_tokens": 384,
    "support_prompt_order": "anchor_then_node_doc_order",
    "support_score_cache_path": "data/cache/reader_support_cache.json",
}


def _param(params: dict[str, Any], key: str) -> Any:
    value = params.get(key)
    return _DEFAULTS[key] if value is None else value


def _to_selected_chunk(unit: EvidenceUnit, chunk_lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Raw chunk -> real chunk (keeps chunk-id recall); else a source-extractive pseudo-chunk."""
    if unit.unit_type == "chunk" and unit.primary_chunk_id in chunk_lookup:
        return dict(chunk_lookup[unit.primary_chunk_id])
    return {
        "chunk_id": f"reader_support::{unit.unit_id}",
        "global_index": unit.global_index_start
        if unit.global_index_start is not None
        else -1,
        "act": unit.act if unit.act is not None else 0,
        "scene": unit.scene if unit.scene is not None else 0,
        "scene_id": unit.scene_id or "reader_support",
        "scene_title": unit.scene_title or "reader_support evidence unit",
        "chunk_in_scene": 0,
        "start_token": 0,
        "end_token": unit.token_count,
        "token_count": unit.token_count,
        "text": unit.text,
        "source_chunk_ids": list(unit.source_chunk_ids),
        "member_chunk_ids": list(unit.metadata.get("member_chunk_ids", unit.source_chunk_ids)),
        "unit_type": unit.unit_type,
    }


def _empty_result(
    request: ContextAssemblyRequest, trace: dict[str, Any], reason: str
) -> ContextAssemblyResult:
    trace["selection_steps"] = []
    trace["final_selected_units"] = []
    trace["final_token_count"] = 0
    trace["empty_reason"] = reason
    return ContextAssemblyResult(
        selected_chunk_ids=[],
        selected_chunks=[],
        original_hit_chunk_ids=[str(row["chunk_id"]) for row in request.retrieval_trace],
        retrieval_trace=[dict(row) for row in request.retrieval_trace],
        retrieval_method="reader_support_empty",
        prompt_order=str(trace.get("prompt_order_strategy", "reader_support")),
        context_assembly_trace=trace,
    )


def assemble_reader_support(request: ContextAssemblyRequest) -> ContextAssemblyResult:
    if not request.retrieval_trace:
        raise ValueError("reader_support requires a dense retrieval trace")
    if request.selector_model is None:
        raise ValueError("reader_support requires the reader/selector model")

    params = request.feature_params
    chunk_lookup = request.chunk_lookup
    question_text = request.question.question
    cache_path = Path(str(_param(params, "support_score_cache_path")))
    prompt_order = str(_param(params, "support_prompt_order"))

    trace: dict[str, Any] = {
        "method": "reader_support",
        "source": "reader-supervised evidence support assembler",
        "deviations": list(READER_SUPPORT_DEVIATIONS),
        "prompt_order_strategy": prompt_order,
    }

    # --- Stage 1: evidence node induction -------------------------------------
    catalog = build_candidate_catalog(
        request.retrieval_trace,
        chunk_lookup,
        int(_param(params, "support_node_candidate_catalog_k")),
    )
    node_cache = JsonKVCache(cache_path, section="node_induction")
    induction = induce_nodes(
        question_text,
        catalog,
        request.selector_model,
        node_cache,
        max_nodes=int(_param(params, "support_max_nodes")),
        max_tokens=int(_param(params, "support_node_induction_max_tokens")),
    )
    nodes: list[EvidenceNode] = induction["nodes"]
    trace["node_induction_prompt"] = induction["prompt"]
    trace["node_induction_raw_output"] = induction["raw_output"]
    trace["node_induction_parse_error"] = induction["parse_error"]
    trace["node_induction_fallback"] = induction["fallback"]
    trace["node_induction_cache_hit"] = induction["cache_hit"]
    trace["nodes"] = [node.to_dict() for node in nodes]

    # --- Stage 2: candidate evidence units ------------------------------------
    unit_types = [
        t.strip()
        for t in str(_param(params, "support_unit_types")).split(",")
        if t.strip()
    ]
    built = build_units(
        request.retrieval_trace,
        chunk_lookup,
        candidate_chunks=int(_param(params, "support_candidate_chunks")),
        unit_types=unit_types,
        include_neighbors=bool(_param(params, "support_include_neighbors")),
        neighbor_hops=int(_param(params, "support_neighbor_hops")),
        max_unit_tokens=int(_param(params, "support_max_unit_tokens")),
        max_units_total=int(_param(params, "support_max_units_total")),
    )
    units: list[EvidenceUnit] = built["units"]
    units_by_id = {unit.unit_id: unit for unit in units}
    trace["generated_units"] = {
        "num_candidate_chunks": built["num_candidate_chunks"],
        "unit_type_counts": built["unit_type_counts"],
        "dropped": built["dropped"],
        "total_units": len(units),
    }
    if not units:
        return _empty_result(request, trace, "no candidate units constructed")

    # --- Stage 3: reader-teacher support scoring (prefiltered) -----------------
    teacher_cache = JsonKVCache(cache_path, section="support_scores")
    scorer = ReaderTeacherSupportScorer(
        request.selector_model,
        teacher_cache,
        max_tokens=int(_param(params, "support_teacher_max_tokens")),
    )
    units_per_node = int(_param(params, "support_teacher_units_per_node"))
    matrix: dict[tuple[str, str], float] = {}
    all_scores: list[SupportScore] = []
    parse_errors = 0
    validation_warnings = 0
    scored_pairs: set[tuple[str, str]] = set()
    per_node_trace: dict[str, Any] = {}
    for node in nodes:
        ranked = sorted(units, key=lambda u: lexical_prior(node, u), reverse=True)
        prefiltered = ranked[: max(1, units_per_node)]
        node_rows: list[dict[str, Any]] = []
        for unit in prefiltered:
            if (node.node_id, unit.unit_id) in scored_pairs:
                continue
            scored_pairs.add((node.node_id, unit.unit_id))
            support = scorer.score(question_text, node, unit)
            matrix[(node.node_id, unit.unit_id)] = support.support_score
            all_scores.append(support)
            if support.parse_error:
                parse_errors += 1
            validation_warnings += len(support.validation_warnings)
            node_rows.append(
                {
                    "unit_id": unit.unit_id,
                    "support_score": support.support_score,
                    "support_type": support.support_type,
                    "lexical_prior": round(lexical_prior(node, unit), 4),
                }
            )
        node_rows.sort(key=lambda r: r["support_score"], reverse=True)
        per_node_trace[node.node_id] = node_rows[:units_per_node]
    trace["support_scoring"] = {
        "pairs_scored": len(all_scores),
        "parse_errors": parse_errors,
        "validation_warnings": validation_warnings,
        "cache_hits": sum(1 for s in all_scores if s.cache_hit),
        "top_units_per_node": per_node_trace,
    }
    # Teacher labels for future training.
    trace["teacher_labels"] = [s.to_dict() for s in all_scores]

    # --- Stage 4: budgeted greedy selection -----------------------------------
    beta = float(_param(params, "support_redundancy_beta"))
    tau = float(_param(params, "support_token_exponent_tau"))
    min_unit_score = float(_param(params, "support_min_unit_score"))
    coverage_threshold = float(_param(params, "support_node_coverage_threshold"))
    max_selected = int(_param(params, "support_max_selected_units"))

    def unit_score(node_id: str, unit_id: str) -> float:
        return matrix.get((node_id, unit_id), 0.0)

    greedy = greedy_select(
        units,
        nodes,
        matrix,
        context_budget=request.context_budget,
        beta=beta,
        tau=tau,
        min_unit_score=min_unit_score,
        coverage_threshold=coverage_threshold,
        max_selected=max_selected,
    )
    if not greedy["selectable"]:
        return _empty_result(
            request, trace, "no candidate unit reached support_min_unit_score"
        )
    selected = greedy["selected"]
    selectable = greedy["selectable"]
    selection_steps = greedy["steps"]
    coverage_progress = greedy["progress"]
    miss = greedy["miss"]
    remaining_budget = greedy["remaining_budget"]

    # --- Optional context repair (needs_more_context) -------------------------
    repairs = _context_repair(
        selected,
        selectable,
        units_by_id,
        all_scores,
        nodes,
        matrix,
        remaining_budget,
    )
    for unit, detail in repairs:
        selected.append(unit)
        remaining_budget -= unit.token_count
        for node in nodes:
            miss[node.node_id] *= 1.0 - unit_score(node.node_id, unit.unit_id)
        selection_steps.append(detail)
        coverage_progress.append({n.node_id: 1.0 - miss[n.node_id] for n in nodes})

    if not selected:
        return _empty_result(request, trace, "greedy selection produced no positive-gain unit")

    # --- Ordering -------------------------------------------------------------
    ordered = _order_units(selected, nodes, matrix, prompt_order)

    final_coverage = {n.node_id: round(1.0 - miss[n.node_id], 4) for n in nodes}
    covered_above = sum(
        1 for n in nodes if (1.0 - miss[n.node_id]) >= coverage_threshold
    )
    trace["coverage"] = {
        "progress": coverage_progress,
        "final_per_node": final_coverage,
        "final_node_coverage_mean": round(
            sum(final_coverage.values()) / len(final_coverage), 4
        ),
        "final_node_coverage_min": min(final_coverage.values()),
        "num_nodes_covered_above_threshold": covered_above,
        "coverage_threshold": coverage_threshold,
    }
    trace["selection_steps"] = selection_steps
    trace["final_selected_units"] = [unit.summary() for unit in ordered]

    selected_chunks = [_to_selected_chunk(unit, chunk_lookup) for unit in ordered]
    selected_ids = [str(chunk["chunk_id"]) for chunk in selected_chunks]
    final_tokens = sum(int(chunk["token_count"]) for chunk in selected_chunks)
    trace["final_token_count"] = final_tokens
    trace["final_ordering"] = selected_ids
    trace["unit_type_counts_selected"] = _count_types(ordered)

    return ContextAssemblyResult(
        selected_chunk_ids=selected_ids,
        selected_chunks=selected_chunks,
        original_hit_chunk_ids=[str(row["chunk_id"]) for row in request.retrieval_trace],
        retrieval_trace=[dict(row) for row in request.retrieval_trace],
        retrieval_method="reader_support",
        prompt_order=prompt_order,
        context_assembly_trace=trace,
    )


def _context_repair(
    selected: list[EvidenceUnit],
    selectable: list[EvidenceUnit],
    units_by_id: dict[str, EvidenceUnit],
    all_scores: list[SupportScore],
    nodes: list[EvidenceNode],
    matrix: dict[tuple[str, str], float],
    remaining_budget: int,
) -> list[tuple[EvidenceUnit, dict[str, Any]]]:
    """If a selected small unit asked for more context, add a fitting neighbor."""
    needs_more = {
        s.unit_id
        for s in all_scores
        if s.needs_more_context and s.support_score > 0.0
    }
    selected_ids = {unit.unit_id for unit in selected}
    by_primary: dict[str, list[EvidenceUnit]] = {}
    for unit in selectable:
        by_primary.setdefault(unit.primary_chunk_id, []).append(unit)

    repairs: list[tuple[EvidenceUnit, dict[str, Any]]] = []
    used_budget = 0
    for unit in selected:
        if unit.unit_id not in needs_more:
            continue
        if unit.unit_type not in {"sentence", "line_span"}:
            continue
        neighbors = [
            candidate
            for candidate in by_primary.get(unit.primary_chunk_id, [])
            if candidate.unit_type.startswith("neighbor")
            and candidate.unit_id not in selected_ids
        ]
        for neighbor in neighbors:
            if neighbor.token_count > (remaining_budget - used_budget):
                continue
            cov_gain = sum(
                matrix.get((n.node_id, neighbor.unit_id), 0.0) for n in nodes
            )
            if cov_gain <= 0.0:
                continue
            used_budget += neighbor.token_count
            selected_ids.add(neighbor.unit_id)
            repairs.append(
                (
                    neighbor,
                    {
                        "unit_id": neighbor.unit_id,
                        "unit_type": neighbor.unit_type,
                        "token_count": neighbor.token_count,
                        "context_repair": True,
                        "repaired_for_unit": unit.unit_id,
                        "coverage_gain": round(cov_gain, 5),
                    },
                )
            )
            break
    return repairs


def _strongest_node(
    unit: EvidenceUnit, nodes: list[EvidenceNode], matrix: dict[tuple[str, str], float]
) -> EvidenceNode:
    return max(
        nodes,
        key=lambda n: matrix.get((n.node_id, unit.unit_id), 0.0),
    )


def _order_units(
    selected: list[EvidenceUnit],
    nodes: list[EvidenceNode],
    matrix: dict[tuple[str, str], float],
    prompt_order: str,
) -> list[EvidenceUnit]:
    def doc_key(unit: EvidenceUnit) -> tuple:
        return (tuple(unit.source_order_key), unit.token_count, unit.unit_id)

    def node_doc_key(unit: EvidenceUnit) -> tuple:
        return (
            _strongest_node(unit, nodes, matrix).order_index,
            tuple(unit.source_order_key),
            unit.token_count,
            unit.unit_id,
        )

    if prompt_order == "document_order":
        return sorted(selected, key=doc_key)
    if prompt_order == "node_doc_order":
        return sorted(selected, key=node_doc_key)
    # Default: anchor_then_node_doc_order
    anchor = max(
        selected,
        key=lambda u: (
            max((matrix.get((n.node_id, u.unit_id), 0.0) for n in nodes), default=0.0),
            -tuple(u.source_order_key)[0] if u.source_order_key else 0,
        ),
    )
    rest = sorted((u for u in selected if u is not anchor), key=node_doc_key)
    return [anchor, *rest]


def _count_types(units: list[EvidenceUnit]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for unit in units:
        counts[unit.unit_type] = counts.get(unit.unit_type, 0) + 1
    return counts
