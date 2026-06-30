"""Procedure contract for the dynamic treatment.

The planner emits a JSON contract describing *how* to gather evidence. This
module parses and **normalizes** it: unknown/missing policy fields fall back to
the run defaults, every correction is logged in `deviations` (never silently
swallowed), and a totally unparseable contract degrades to a single-node,
parallel default so a run never crashes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from hamlet_qa.core.evidence.schema import EvidenceNode

RETRIEVAL_MODES = {"parallel", "sequential"}
SELECTION_POLICIES = {"greedy_coverage", "top_per_node"}
ORDERING_POLICIES = {"document_order", "node_order", "anchor_first"}
SUPPORT_POLICIES = {"reranker", "teacher"}
QUESTION_TYPES = {"single", "independent_multipart", "bridge_multihop"}

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ProcedureContract:
    question_type: str
    retrieval_policy: str
    retrieval_mode: str
    selection_policy: str
    ordering_policy: str
    support_policy: str
    nodes: list[EvidenceNode]
    strategy: str = ""
    deviations: list[str] = field(default_factory=list)
    parse_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_type": self.question_type,
            "retrieval_policy": self.retrieval_policy,
            "retrieval_mode": self.retrieval_mode,
            "selection_policy": self.selection_policy,
            "ordering_policy": self.ordering_policy,
            "support_policy": self.support_policy,
            "strategy": self.strategy,
            "nodes": [node.to_dict() for node in self.nodes],
            "deviations": list(self.deviations),
            "parse_error": self.parse_error,
        }


def nodes_from_items(items: Any, max_nodes: int) -> list[EvidenceNode]:
    """Parse a JSON node list (shared shape) into EvidenceNode objects."""
    if not isinstance(items, list):
        return []
    nodes: list[EvidenceNode] = []
    for index, item in enumerate(items[:max_nodes], start=1):
        if not isinstance(item, dict):
            continue
        need = str(item.get("need", "")).strip()
        if not need:
            continue
        node_query = str(item.get("node_query", "")).strip() or need
        try:
            order_index = int(item.get("order_index", index))
        except (TypeError, ValueError):
            order_index = index
        depends_on = [str(d) for d in item.get("depends_on", []) if str(d).strip()]
        nodes.append(
            EvidenceNode(
                node_id=str(item.get("node_id") or f"n{index}"),
                need=need,
                node_query=node_query,
                order_index=order_index,
                depends_on=depends_on,
                raw_reason=str(item.get("reason", "")).strip(),
            )
        )
    return nodes


def _normalize(value: Any, allowed: set[str], default: str, label: str, deviations: list[str]) -> str:
    text = str(value or "").strip().lower()
    if text in allowed:
        return text
    if value not in (None, ""):
        deviations.append(f"{label}='{value}' is invalid; using default '{default}'")
    return default


def parse_contract(
    raw_output: str,
    *,
    defaults: dict[str, str],
    max_nodes: int,
    question_text: str,
) -> ProcedureContract:
    """Validate/normalize the planner output against the run defaults."""
    deviations: list[str] = []
    parse_error: str | None = None
    match = _JSON_OBJECT.search(raw_output or "")
    parsed: dict[str, Any] = {}
    if match is None:
        parse_error = "no JSON object found"
    else:
        try:
            loaded = json.loads(match.group(0))
            parsed = loaded if isinstance(loaded, dict) else {}
        except json.JSONDecodeError as error:
            parse_error = f"json decode error: {error}"

    nodes = nodes_from_items(parsed.get("nodes"), max_nodes)
    if not nodes:
        if not parse_error:
            parse_error = "contract had no usable nodes"
        deviations.append("no parseable nodes; falling back to one whole-question node")
        nodes = [
            EvidenceNode(
                node_id="n1",
                need=f"Find all source text needed to answer: {question_text}",
                node_query=question_text,
                order_index=1,
            )
        ]

    retrieval_policy = _normalize(
        parsed.get("retrieval_policy"), {"dense"}, "dense", "retrieval_policy", deviations
    )
    return ProcedureContract(
        question_type=_normalize(
            parsed.get("question_type"), QUESTION_TYPES, "single", "question_type", deviations
        ),
        retrieval_policy=retrieval_policy,
        retrieval_mode=_normalize(
            parsed.get("retrieval_mode"), RETRIEVAL_MODES, defaults["retrieval_mode"],
            "retrieval_mode", deviations,
        ),
        selection_policy=_normalize(
            parsed.get("selection_policy"), SELECTION_POLICIES, defaults["selection_policy"],
            "selection_policy", deviations,
        ),
        ordering_policy=_normalize(
            parsed.get("ordering_policy"), ORDERING_POLICIES, defaults["ordering_policy"],
            "ordering_policy", deviations,
        ),
        support_policy=_normalize(
            parsed.get("support_policy"), SUPPORT_POLICIES, defaults["support_policy"],
            "support_policy", deviations,
        ),
        nodes=nodes,
        strategy=str(parsed.get("strategy", "")).strip(),
        deviations=deviations,
        parse_error=parse_error,
    )
