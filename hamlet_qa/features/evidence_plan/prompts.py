"""Swappable prompt registries for the evidence-planning experiment.

Three prompt "slots", each a named registry so a single CLI flag selects the
active variant (the experimental knob):

- DECOMPOSITION (`--decomp-prompt`)  — used by `plan_fixed` to break the
  question into evidence nodes. The variants differ only in the *cognitive
  scaffolding* given to the model (just sub-questions vs. information
  requirements vs. an explicit step-wise strategy) so we can measure which
  decomposition style helps.
- PLANNER (`--planner-prompt`)       — used by `plan_dynamic` to emit a full
  procedure contract (nodes + retrieval/selection/ordering policies).
- FOLLOW-UP (`--followup-prompt`)    — used in sequential retrieval to rewrite a
  dependent node's search query from the evidence gathered so far.

Every variant carries a version string baked into LLM cache keys, so editing a
prompt never reads a stale cache. All decomposition / planner node lists use the
SAME JSON node shape (`need`, `node_query`, ...) so parsing is shared.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptVariant:
    name: str
    version: str
    system: str
    template: str  # str.format fields: {question} {catalog} {max_nodes} (+ slot-specific)


# The required JSON node shape, shared by every decomposition/planner variant.
_NODE_SHAPE = (
    '{{"node_id": "n1", "need": "<the sub-question to answer>", '
    '"node_query": "<a search query to find evidence for it>", '
    '"order_index": 1, "depends_on": [], "reason": "<why>"}}'
)

# ---------------------------------------------------------------------------
# Slot A: decomposition (plan_fixed)
# ---------------------------------------------------------------------------

_DECOMP_SYSTEM = (
    "You are an expert reading-comprehension analyst. You break a question about "
    "a long document into the distinct information needs that must each be "
    "supported by source text before the question can be answered. You describe "
    "needs; you never write or guess the answer."
)

_DECOMP_COMMON_TAIL = (
    "\n\nQUESTION:\n{question}\n\n"
    "CANDIDATE CATALOG (retrieved passages, for topical context only — do not "
    "assume it contains the answers):\n{catalog}\n\n"
    "Respond with JSON ONLY in exactly this shape:\n"
    '{{"nodes": [' + _NODE_SHAPE + "]}}"
)

DECOMPOSITION_PROMPTS: dict[str, PromptVariant] = {
    "split_questions": PromptVariant(
        name="split_questions",
        version="decomp.split_questions.v1",
        system=_DECOMP_SYSTEM,
        template=(
            "Break the QUESTION into at most {max_nodes} answerable sub-questions. "
            "Each sub-question is one thing that must be looked up. Keep them "
            "non-overlapping; split only genuinely separate needs."
            + _DECOMP_COMMON_TAIL
        ),
    ),
    "list_requirements": PromptVariant(
        name="list_requirements",
        version="decomp.list_requirements.v1",
        system=_DECOMP_SYSTEM,
        template=(
            "List at most {max_nodes} INFORMATION REQUIREMENTS for the QUESTION: "
            "the specific facts, entities, relations, or events that the answer "
            "depends on. For each requirement, give the need and a focused search "
            "query (include the key entity/relation terms)."
            + _DECOMP_COMMON_TAIL
        ),
    ),
    "reason_then_plan": PromptVariant(
        name="reason_then_plan",
        version="decomp.reason_then_plan.v1",
        system=_DECOMP_SYSTEM,
        template=(
            "First THINK step by step about how to solve the QUESTION: what kind "
            "of question is it, what must be found, in what order, and which parts "
            "depend on earlier parts. Write this as a short numbered strategy. "
            "THEN turn the strategy into at most {max_nodes} evidence nodes; use "
            "order_index and depends_on to encode the order and dependencies from "
            "your strategy."
            + _DECOMP_COMMON_TAIL.replace(
                'Respond with JSON ONLY in exactly this shape:\n{{"nodes": [',
                "Respond with JSON ONLY in exactly this shape (put your numbered "
                'strategy in the "strategy" field):\n'
                '{{"strategy": "<numbered plan>", "nodes": [',
            )
        ),
    ),
}

# ---------------------------------------------------------------------------
# Slot B: planner (plan_dynamic) — emits a full procedure contract
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = (
    "You are a retrieval-strategy planner. Given a question about a long "
    "document, you design the procedure to gather its evidence: how to split it, "
    "whether to retrieve the parts in parallel or step by step, how to select, "
    "and how to order. You output a single JSON procedure contract. You never "
    "write the answer."
)

_CONTRACT_SHAPE = (
    "{{\n"
    '  "question_type": "single | independent_multipart | bridge_multihop",\n'
    '  "retrieval_policy": "dense",\n'
    '  "retrieval_mode": "parallel | sequential",\n'
    '  "selection_policy": "greedy_coverage | top_per_node",\n'
    '  "ordering_policy": "document_order | node_order | anchor_first",\n'
    '  "support_policy": "reranker | teacher",\n'
    '  "nodes": [' + _NODE_SHAPE + "]\n"
    "}}"
)

_PLANNER_COMMON_TAIL = (
    "\n\nGuidance:\n"
    "- Use retrieval_mode \"sequential\" only when a later node genuinely needs "
    "the answer of an earlier node to be retrievable (a bridge); otherwise use "
    "\"parallel\". Encode dependencies with depends_on.\n"
    "- Use selection_policy \"greedy_coverage\" to cover every node compactly; "
    "\"top_per_node\" to just keep each node's best passages.\n\n"
    "QUESTION:\n{question}\n\n"
    "CANDIDATE CATALOG (topical context only):\n{catalog}\n\n"
    "Respond with JSON ONLY in exactly this shape:\n" + _CONTRACT_SHAPE
)

PLANNER_PROMPTS: dict[str, PromptVariant] = {
    "contract_v1": PromptVariant(
        name="contract_v1",
        version="planner.contract_v1.v1",
        system=_PLANNER_SYSTEM,
        template=(
            "Design a procedure to answer the QUESTION over the document, with at "
            "most {max_nodes} nodes." + _PLANNER_COMMON_TAIL
        ),
    ),
    "strategy_contract": PromptVariant(
        name="strategy_contract",
        version="planner.strategy_contract.v1",
        system=_PLANNER_SYSTEM,
        template=(
            "First write a short numbered STRATEGY (the mental algorithm) for "
            "solving the QUESTION: what to find, in what order, and what depends "
            "on what. Then encode that strategy as the procedure contract with at "
            "most {max_nodes} nodes (put the strategy text in a top-level "
            '"strategy" field).' + _PLANNER_COMMON_TAIL.replace(
                "Respond with JSON ONLY in exactly this shape:\n{{\n",
                'Respond with JSON ONLY in exactly this shape:\n{{\n  "strategy": '
                '"<numbered plan>",\n',
            )
        ),
    ),
}

# ---------------------------------------------------------------------------
# Slot C: follow-up query reformulation (sequential retrieval)
# ---------------------------------------------------------------------------

_FOLLOWUP_SYSTEM = (
    "You rewrite a search query for one step of a multi-step lookup. You use only "
    "the evidence gathered so far to make the next query specific (for example by "
    "filling in an entity that earlier evidence revealed). You never answer the "
    "overall question."
)

_FOLLOWUP_COMMON_TAIL = (
    "\n\nORIGINAL QUESTION:\n{question}\n\n"
    "THIS STEP'S NEED:\n{need}\n\n"
    "EVIDENCE GATHERED SO FAR:\n{evidence}\n\n"
    'Respond with JSON ONLY: {{"retrieval_query": "<the rewritten search query>"}}'
)

FOLLOWUP_PROMPTS: dict[str, PromptVariant] = {
    "rewrite_with_evidence": PromptVariant(
        name="rewrite_with_evidence",
        version="followup.rewrite_with_evidence.v1",
        system=_FOLLOWUP_SYSTEM,
        template=(
            "Rewrite the search query for THIS STEP'S NEED so it incorporates any "
            "specific entity, name, or fact revealed by the evidence so far."
            + _FOLLOWUP_COMMON_TAIL
        ),
    ),
}


def get_decomposition_prompt(name: str) -> PromptVariant:
    return _get(DECOMPOSITION_PROMPTS, name, "decomposition")


def get_planner_prompt(name: str) -> PromptVariant:
    return _get(PLANNER_PROMPTS, name, "planner")


def get_followup_prompt(name: str) -> PromptVariant:
    return _get(FOLLOWUP_PROMPTS, name, "follow-up")


def _get(registry: dict[str, PromptVariant], name: str, slot: str) -> PromptVariant:
    try:
        return registry[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown {slot} prompt '{name}'. Available: {sorted(registry)}"
        ) from error
