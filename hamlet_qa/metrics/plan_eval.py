"""Stage-wise evaluation for the evidence-planning treatments (reader-free).

The experiment is judged at three points; this module measures the two that
need no LLM, straight from gold chunk IDs and the logged plan trace:

1. **After the plan is generated** — were the evidence slots (nodes) produced?
   Structural only here (count + fallback); whether they are the *right* slots
   is left to the future LLM judge, which attaches to the logged
   `context_assembly_trace[...].nodes`.
2. **After retrieval** — did the per-node ("evidence-slot") retrieval actually
   surface the gold chunk for each required evidence role? Fully measurable:
   a role is "retrieved" if any of its gold chunks appears in the union of the
   nodes' retrieved candidates. This is the headline metric for the experiment.
3. **After the final answer** — generation quality. Intentionally NOT scored
   here; the stronger-model LLM judge will attach to the row's `model_output`.

Only the `plan_fixed` / `plan_dynamic` treatments carry per-node retrieval, so
the metric returns nulls (with `plan_eval_applicable=False`) for other rows.
"""

from __future__ import annotations

from typing import Any


def _plan_nodes(trace: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Return (generated nodes, fallback flag) for either plan treatment."""
    if "decomposition" in trace:  # plan_fixed
        gen = trace.get("decomposition") or {}
        return list(gen.get("nodes") or []), bool(gen.get("fallback"))
    contract = trace.get("contract") or {}  # plan_dynamic
    return list(contract.get("nodes") or []), bool(contract.get("parse_error"))


def _gold_by_role(row: dict[str, Any]) -> dict[str, set[str]]:
    role_gold: dict[str, set[str]] = {}
    for index, quote in enumerate(row.get("required_evidence_quotes") or []):
        role = str(quote.get("role") or f"quote_{index}")
        role_gold.setdefault(role, set()).update(
            str(cid) for cid in (quote.get("matched_chunk_ids") or [])
        )
    return role_gold


def _none_result() -> dict[str, Any]:
    return {
        "plan_eval_applicable": False,
        "plan_num_nodes": None,
        "plan_node_fallback": None,
        "plan_num_gold_roles": None,
        "plan_slot_retrieval_recall": None,
        "plan_gold_chunk_retrieval": None,
        "plan_slot_detail": None,
    }


def compute_plan_eval_for_row(row: dict[str, Any]) -> dict[str, Any]:
    trace = row.get("context_assembly_trace") or {}
    if trace.get("method") not in ("plan_fixed", "plan_dynamic"):
        return _none_result()

    # --- Stage 1: were the evidence slots generated? --------------------------
    nodes, fallback = _plan_nodes(trace)
    role_gold = _gold_by_role(row)
    num_gold_roles = len(role_gold)

    # --- Stage 2: did per-node retrieval surface each slot's gold chunk? ------
    per_node = (trace.get("execution") or {}).get("per_node_retrieval") or []
    node_retrieved: dict[str, set[str]] = {}
    for entry in per_node:
        node_id = str(entry.get("node_id"))
        node_retrieved[node_id] = {
            str(r.get("chunk_id")) for r in (entry.get("retrieved") or [])
        }
    union_retrieved: set[str] = set().union(*node_retrieved.values()) if node_retrieved else set()

    slot_detail: list[dict[str, Any]] = []
    covered = 0
    for role, gold_ids in role_gold.items():
        retrieved_by = sorted(nid for nid, got in node_retrieved.items() if gold_ids & got)
        is_retrieved = bool(gold_ids & union_retrieved)
        covered += int(is_retrieved)
        slot_detail.append(
            {
                "role": role,
                "gold_chunk_ids": sorted(gold_ids),
                "retrieved": is_retrieved,
                "retrieved_by_nodes": retrieved_by,
            }
        )

    all_gold = {str(cid) for cid in (row.get("derived_gold_chunk_ids") or [])}
    return {
        "plan_eval_applicable": True,
        # Stage 1 — plan generation (structural; semantic role match -> LLM judge)
        "plan_num_nodes": len(nodes),
        "plan_node_fallback": fallback,
        "plan_num_gold_roles": num_gold_roles,
        # Stage 2 — retrieval through evidence slots (measured against gold)
        "plan_slot_retrieval_recall": (covered / num_gold_roles) if num_gold_roles else None,
        "plan_gold_chunk_retrieval": (
            len(all_gold & union_retrieved) / len(all_gold) if all_gold else None
        ),
        "plan_slot_detail": slot_detail,
        # Stage 3 — final-answer quality is scored later by the LLM judge, which
        # reads row["model_output"]; nothing computed here.
    }
