"""LLM calls for the planning stage: decompose, plan, and reformulate.

All three are cached (`JsonKVCache`) and keyed by the active prompt *variant +
version*, so swapping `--decomp-prompt` / `--planner-prompt` / `--followup-prompt`
never reuses another variant's cache. Parsing degrades gracefully and records
the failure rather than raising.
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any

from hamlet_qa.core.llm_cache import JsonKVCache, stable_hash
from hamlet_qa.features.evidence_plan.contract import (
    ProcedureContract,
    nodes_from_items,
    parse_contract,
)
from hamlet_qa.features.evidence_plan.prompts import (
    PromptVariant,
    get_decomposition_prompt,
    get_followup_prompt,
    get_planner_prompt,
)
from hamlet_qa.core.evidence.schema import EvidenceNode

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _call_model(model: Any, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    signature = inspect.signature(model.generate)
    if "max_tokens" in signature.parameters:
        return str(model.generate(system_prompt, user_prompt, max_tokens=max_tokens))
    return str(model.generate(system_prompt, user_prompt))


def _extract_json(raw_output: str) -> dict[str, Any] | None:
    match = _JSON_OBJECT.search(raw_output or "")
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def decompose(
    question_text: str,
    model: Any,
    variant: PromptVariant,
    cache: JsonKVCache,
    *,
    max_nodes: int,
    max_tokens: int,
) -> dict[str, Any]:
    """Run a decomposition prompt -> evidence nodes (+ optional strategy text).

    The prompt sees only the question (no retrieved preview), so this measures the
    reader's ability to decompose from the question text alone.
    """
    prompt = variant.template.format(question=question_text, max_nodes=max_nodes)
    model_name = str(getattr(model, "model_name", "reader_model"))
    cache_key = stable_hash(
        {"prompt": prompt, "model": model_name, "variant": variant.name, "version": variant.version}
    )
    cached = cache.get(cache_key)
    if cached is not None and cached.get("nodes") is not None:
        return {
            "nodes": [EvidenceNode(**node) for node in cached["nodes"]],
            "strategy": cached.get("strategy", ""),
            "prompt": prompt,
            "raw_output": cached.get("raw_output", ""),
            "parse_error": cached.get("parse_error"),
            "fallback": cached.get("fallback", False),
            "prompt_variant": variant.name,
            "cache_hit": True,
        }

    raw_output = _call_model(model, variant.system, prompt, max_tokens)
    parsed = _extract_json(raw_output)
    nodes = nodes_from_items(parsed.get("nodes") if parsed else None, max_nodes)
    strategy = str(parsed.get("strategy", "")).strip() if parsed else ""
    parse_error: str | None = None
    fallback = False
    if not nodes:
        parse_error = "decomposition output had no usable nodes"
        fallback = True
        nodes = [
            EvidenceNode(
                node_id="n1",
                need=f"Find all source text needed to answer: {question_text}",
                node_query=question_text,
                order_index=1,
            )
        ]
    cache.set(
        cache_key,
        {
            "nodes": [node.to_dict() for node in nodes],
            "strategy": strategy,
            "raw_output": raw_output,
            "parse_error": parse_error,
            "fallback": fallback,
            "version": variant.version,
        },
    )
    cache.save()
    return {
        "nodes": nodes,
        "strategy": strategy,
        "prompt": prompt,
        "raw_output": raw_output,
        "parse_error": parse_error,
        "fallback": fallback,
        "prompt_variant": variant.name,
        "cache_hit": False,
    }


def plan(
    question_text: str,
    model: Any,
    variant: PromptVariant,
    cache: JsonKVCache,
    *,
    defaults: dict[str, str],
    max_nodes: int,
    max_tokens: int,
) -> dict[str, Any]:
    """Run a planner prompt -> a validated/normalized ProcedureContract.

    The prompt sees only the question (no retrieved preview).
    """
    prompt = variant.template.format(question=question_text, max_nodes=max_nodes)
    model_name = str(getattr(model, "model_name", "reader_model"))
    cache_key = stable_hash(
        {
            "prompt": prompt,
            "model": model_name,
            "variant": variant.name,
            "version": variant.version,
            "defaults": defaults,
        }
    )
    cached = cache.get(cache_key)
    if cached is not None and cached.get("contract") is not None:
        contract = _contract_from_dict(cached["contract"])
        return {
            "contract": contract,
            "prompt": prompt,
            "raw_output": cached.get("raw_output", ""),
            "prompt_variant": variant.name,
            "cache_hit": True,
        }

    raw_output = _call_model(model, variant.system, prompt, max_tokens)
    contract = parse_contract(
        raw_output, defaults=defaults, max_nodes=max_nodes, question_text=question_text
    )
    cache.set(
        cache_key,
        {"contract": contract.to_dict(), "raw_output": raw_output, "version": variant.version},
    )
    cache.save()
    return {
        "contract": contract,
        "prompt": prompt,
        "raw_output": raw_output,
        "prompt_variant": variant.name,
        "cache_hit": False,
    }


def _contract_from_dict(data: dict[str, Any]) -> ProcedureContract:
    return ProcedureContract(
        question_type=data["question_type"],
        retrieval_policy=data["retrieval_policy"],
        retrieval_mode=data["retrieval_mode"],
        selection_policy=data["selection_policy"],
        ordering_policy=data["ordering_policy"],
        support_policy=data["support_policy"],
        nodes=[EvidenceNode(**node) for node in data.get("nodes", [])],
        strategy=data.get("strategy", ""),
        deviations=list(data.get("deviations", [])),
        parse_error=data.get("parse_error"),
    )


def reformulate_query(
    question_text: str,
    node: EvidenceNode,
    evidence: str,
    model: Any,
    variant: PromptVariant,
    cache: JsonKVCache,
    *,
    max_tokens: int,
) -> dict[str, Any]:
    """Rewrite a dependent node's search query from gathered evidence (sequential)."""
    prompt = variant.template.format(
        question=question_text, need=node.need, evidence=evidence or "(none yet)"
    )
    model_name = str(getattr(model, "model_name", "reader_model"))
    cache_key = stable_hash(
        {
            "prompt": prompt,
            "model": model_name,
            "variant": variant.name,
            "version": variant.version,
        }
    )
    cached = cache.get(cache_key)
    if cached is not None and "query" in cached:
        return {"query": cached["query"], "raw_output": cached.get("raw_output", ""), "cache_hit": True}

    raw_output = _call_model(model, variant.system, prompt, max_tokens)
    parsed = _extract_json(raw_output)
    query = ""
    if parsed is not None:
        query = str(parsed.get("retrieval_query", "")).strip()
    if not query:
        # Fall back to the first non-empty line, then to the node's own query.
        lines = [line.strip() for line in str(raw_output).splitlines() if line.strip()]
        query = lines[0] if lines else node.node_query
    record = {"query": query, "raw_output": raw_output, "version": variant.version}
    cache.set(cache_key, record)
    cache.save()
    return {"query": query, "raw_output": raw_output, "cache_hit": False}


def select_variants(
    decomp_name: str, planner_name: str, followup_name: str
) -> dict[str, PromptVariant]:
    """Resolve the active prompt variants (raises on unknown names)."""
    return {
        "decomposition": get_decomposition_prompt(decomp_name),
        "planner": get_planner_prompt(planner_name),
        "followup": get_followup_prompt(followup_name),
    }
