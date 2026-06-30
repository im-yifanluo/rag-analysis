"""Stage 1: evidence-node induction.

The reader model decomposes the question into a small set of *information
needs* (evidence nodes), conditioned on the question plus a compact catalog of
retrieved candidate chunks. Node descriptions must state what to find, never
the answer itself. All calls are cached via JsonKVCache.
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any

from hamlet_qa.core.evidence.schema import EvidenceNode
from hamlet_qa.core.llm_cache import JsonKVCache, stable_hash

NODE_PROMPT_VERSION = "reader_support.nodes.v1"

NODE_INDUCTION_SYSTEM = (
    "You are an expert reading-comprehension analyst. You decompose a question "
    "about a long document into the distinct information needs that must each "
    "be supported by source text before the question can be answered."
)

# Note the doubled braces around the JSON skeleton so str.format leaves them
# literal while substituting {question}, {catalog}, and {max_nodes}.
NODE_INDUCTION_TEMPLATE = """Decompose the QUESTION into at most {max_nodes} evidence nodes. An evidence node is ONE information need that must be supported by source text to answer the question.

Rules:
- Describe the information NEED, do not write or guess the answer.
  Good: "Identify the final fate of Rosencrantz and Guildenstern."
  Bad: "They are dead."
- Prefer fewer, non-overlapping nodes. Split only genuinely separate needs (for example, a multi-part question).
- Use the CANDIDATE CATALOG only to understand what the question is asking about. Do not assume the catalog contains the answers.
- order_index reflects the natural reading/answer order of the needs (1, 2, 3, ...).
- depends_on lists node_ids that must be resolved first, or [] if none.

QUESTION:
{question}

CANDIDATE CATALOG (retrieved passages, for topical context only):
{catalog}

Respond with JSON ONLY in exactly this shape:
{{
  "nodes": [
    {{"node_id": "n1", "need": "...", "node_query": "...", "order_index": 1, "depends_on": [], "reason": "..."}}
  ]
}}"""

NODE_REPAIR_SUFFIX = (
    "\n\nYour previous response could not be parsed as JSON. Respond again with "
    "valid JSON ONLY, no prose, matching the required shape exactly."
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def build_node_prompt(question_text: str, catalog: str, max_nodes: int) -> str:
    return NODE_INDUCTION_TEMPLATE.format(
        question=question_text,
        catalog=catalog,
        max_nodes=max_nodes,
    )


def _extract_json(raw_output: str) -> dict[str, Any] | None:
    match = _JSON_OBJECT.search(raw_output or "")
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_nodes(raw_output: str, max_nodes: int) -> list[EvidenceNode] | None:
    """Parse the node-induction JSON; return None on any structural failure."""
    parsed = _extract_json(raw_output)
    if parsed is None or not isinstance(parsed.get("nodes"), list):
        return None
    nodes: list[EvidenceNode] = []
    for index, item in enumerate(parsed["nodes"][:max_nodes], start=1):
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
    return nodes or None


def fallback_node(question_text: str) -> EvidenceNode:
    return EvidenceNode(
        node_id="n1",
        need=f"Find all source text needed to answer: {question_text}",
        node_query=question_text,
        order_index=1,
        depends_on=[],
        raw_reason="fallback: node induction did not return parseable JSON",
    )


def _call_model(model: Any, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    signature = inspect.signature(model.generate)
    if "max_tokens" in signature.parameters:
        return str(model.generate(system_prompt, user_prompt, max_tokens=max_tokens))
    return str(model.generate(system_prompt, user_prompt))


def induce_nodes(
    question_text: str,
    catalog: str,
    selector_model: Any,
    cache: JsonKVCache,
    max_nodes: int,
    max_tokens: int,
) -> dict[str, Any]:
    """Run node induction with one JSON-repair retry and a single-node fallback."""
    prompt = build_node_prompt(question_text, catalog, max_nodes)
    model_name = str(getattr(selector_model, "model_name", "reader_model"))
    cache_key = stable_hash(
        {
            "prompt": prompt,
            "model": model_name,
            "prompt_version": NODE_PROMPT_VERSION,
            "max_nodes": max_nodes,
        }
    )
    cached = cache.get(cache_key)
    if cached is not None and cached.get("nodes") is not None:
        nodes = [EvidenceNode(**node) for node in cached["nodes"]]
        return {
            "nodes": nodes,
            "prompt": prompt,
            "raw_output": cached.get("raw_output", ""),
            "parse_error": cached.get("parse_error"),
            "fallback": cached.get("fallback", False),
            "cache_hit": True,
        }

    raw_output = _call_model(
        selector_model, NODE_INDUCTION_SYSTEM, prompt, max_tokens
    )
    nodes = parse_nodes(raw_output, max_nodes)
    parse_error: str | None = None
    fallback = False
    if nodes is None:
        parse_error = "primary node-induction output was not parseable JSON"
        repair_output = _call_model(
            selector_model,
            NODE_INDUCTION_SYSTEM,
            prompt + NODE_REPAIR_SUFFIX,
            max_tokens,
        )
        raw_output = f"{raw_output}\n---REPAIR---\n{repair_output}"
        nodes = parse_nodes(repair_output, max_nodes)
    if nodes is None:
        parse_error = "node induction failed after repair; using single-node fallback"
        nodes = [fallback_node(question_text)]
        fallback = True

    cache.set(
        cache_key,
        {
            "nodes": [node.to_dict() for node in nodes],
            "raw_output": raw_output,
            "parse_error": parse_error,
            "fallback": fallback,
            "prompt_version": NODE_PROMPT_VERSION,
        },
    )
    cache.save()
    return {
        "nodes": nodes,
        "prompt": prompt,
        "raw_output": raw_output,
        "parse_error": parse_error,
        "fallback": fallback,
        "cache_hit": False,
    }
