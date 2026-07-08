"""Assembly adapters for the two planning treatments.

`plan_fixed` decomposes with a swappable prompt and runs the executor with
policies taken from the run flags. `plan_dynamic` runs a planner prompt, parses
the procedure contract, and runs the *same* executor with the contract's
policies. Everything heavy (per-node retrieval, support scoring, greedy
coverage) is shared in `executor.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hamlet_qa.core.context import ContextAssemblyRequest, ContextAssemblyResult
from hamlet_qa.core.llm_cache import JsonKVCache
from hamlet_qa.features.evidence_plan.executor import execute_plan
from hamlet_qa.features.evidence_plan.planning import decompose, plan, reformulate_query
from hamlet_qa.features.evidence_plan.prompts import (
    get_decomposition_prompt,
    get_followup_prompt,
    get_planner_prompt,
)
from hamlet_qa.core.evidence.support_teacher import ReaderTeacherSupportScorer

_DEFAULTS: dict[str, Any] = {
    "plan_decomp_prompt": "list_requirements",
    "plan_planner_prompt": "contract_v1",
    "plan_followup_prompt": "rewrite_with_evidence",
    "plan_retrieval_mode": "parallel",
    "plan_support_policy": "reranker",
    "plan_selection_policy": "greedy_coverage",
    "plan_ordering_policy": "document_order",
    "plan_node_top_k": 10,
    "plan_max_nodes": 5,
    "plan_min_support": 0.5,
    "plan_support_temp": 1.0,
    "plan_coverage_threshold": 0.85,
    "plan_redundancy_beta": 0.15,
    "plan_token_exponent_tau": 0.7,
    "plan_max_selected_units": 8,
    "plan_llm_max_tokens": 1024,
    "plan_followup_max_tokens": 256,
    "plan_teacher_max_tokens": 384,
    "plan_cache_path": "data/cache/evidence_plan_cache.json",
}


def _p(params: dict[str, Any], key: str) -> Any:
    value = params.get(key)
    return _DEFAULTS[key] if value is None else value


def _shared(request: ContextAssemblyRequest) -> dict[str, Any]:
    if not request.retrieval_trace:
        raise ValueError("evidence_plan treatments require a dense retrieval trace")
    if request.selector_model is None:
        raise ValueError("evidence_plan treatments require the reader/selector model")
    node_retriever = request.feature_handles.get("node_retriever")
    if node_retriever is None:
        raise ValueError(
            "evidence_plan treatments require a per-node retriever "
            "(feature_handles['node_retriever'])"
        )
    params = request.feature_params
    cache_path = Path(str(_p(params, "plan_cache_path")))
    return {
        "params": params,
        "cache_path": cache_path,
        "node_retriever": node_retriever,
        "model": request.selector_model,
        "question_text": request.question.question,
    }


def _build_teacher(request: ContextAssemblyRequest, support_policy: str, cache_path: Path) -> Any:
    if support_policy != "teacher":
        return None
    return ReaderTeacherSupportScorer(
        request.selector_model,
        JsonKVCache(cache_path, section="plan_teacher_scores"),
        max_tokens=int(_p(request.feature_params, "plan_teacher_max_tokens")),
    )


def _build_reformulate(request: ContextAssemblyRequest, cache_path: Path) -> Any:
    params = request.feature_params
    variant = get_followup_prompt(str(_p(params, "plan_followup_prompt")))
    cache = JsonKVCache(cache_path, section="plan_followup")
    max_tokens = int(_p(params, "plan_followup_max_tokens"))

    def reformulate(node: Any, evidence: str) -> dict[str, Any]:
        return reformulate_query(
            request.question.question, node, evidence, request.selector_model,
            variant, cache, max_tokens=max_tokens,
        )

    return reformulate


def _run(
    request: ContextAssemblyRequest,
    ctx: dict[str, Any],
    nodes: list[Any],
    policies: dict[str, str],
    trace: dict[str, Any],
    method: str,
) -> ContextAssemblyResult:
    params = ctx["params"]
    cache_path = ctx["cache_path"]
    result = execute_plan(
        ctx["question_text"],
        nodes,
        node_retriever=ctx["node_retriever"],
        chunk_lookup=request.chunk_lookup,
        context_budget=request.context_budget,
        retrieval_mode=policies["retrieval_mode"],
        support_policy=policies["support_policy"],
        selection_policy=policies["selection_policy"],
        ordering_policy=policies["ordering_policy"],
        node_top_k=int(_p(params, "plan_node_top_k")),
        support_temp=float(_p(params, "plan_support_temp")),
        coverage_threshold=float(_p(params, "plan_coverage_threshold")),
        redundancy_beta=float(_p(params, "plan_redundancy_beta")),
        token_exponent_tau=float(_p(params, "plan_token_exponent_tau")),
        min_support=float(_p(params, "plan_min_support")),
        max_selected_units=int(_p(params, "plan_max_selected_units")),
        teacher_scorer=_build_teacher(request, policies["support_policy"], cache_path),
        reformulate=_build_reformulate(request, cache_path)
        if policies["retrieval_mode"] == "sequential"
        else None,
    )
    trace["execution"] = result["trace"]
    trace["policies"] = policies
    original_hits = [str(row["chunk_id"]) for row in request.retrieval_trace]
    return ContextAssemblyResult(
        selected_chunk_ids=result["selected_chunk_ids"],
        selected_chunks=result["selected_chunks"],
        original_hit_chunk_ids=original_hits,
        retrieval_trace=[dict(row) for row in request.retrieval_trace],
        retrieval_method=method if not result["empty"] else f"{method}_empty",
        prompt_order=policies["ordering_policy"],
        context_assembly_trace=trace,
    )


def assemble_plan_fixed(request: ContextAssemblyRequest) -> ContextAssemblyResult:
    ctx = _shared(request)
    params = ctx["params"]
    variant = get_decomposition_prompt(str(_p(params, "plan_decomp_prompt")))
    decomposition = decompose(
        ctx["question_text"],
        ctx["model"],
        variant,
        JsonKVCache(ctx["cache_path"], section="plan_decomposition"),
        max_nodes=int(_p(params, "plan_max_nodes")),
        max_tokens=int(_p(params, "plan_llm_max_tokens")),
    )
    policies = {
        "retrieval_mode": str(_p(params, "plan_retrieval_mode")),
        "support_policy": str(_p(params, "plan_support_policy")),
        "selection_policy": str(_p(params, "plan_selection_policy")),
        "ordering_policy": str(_p(params, "plan_ordering_policy")),
    }
    trace = {
        "method": "plan_fixed",
        "source": "evidence-planning experiment (fixed procedure)",
        "deviations": ["fixed procedure: policies come from run flags, not the LLM"],
        "decomposition": {
            "prompt_variant": decomposition["prompt_variant"],
            "prompt": decomposition["prompt"],
            "raw_output": decomposition["raw_output"],
            "strategy": decomposition["strategy"],
            "parse_error": decomposition["parse_error"],
            "fallback": decomposition["fallback"],
            "cache_hit": decomposition["cache_hit"],
            "nodes": [node.to_dict() for node in decomposition["nodes"]],
        },
    }
    return _run(request, ctx, decomposition["nodes"], policies, trace, "plan_fixed")


def assemble_plan_dynamic(request: ContextAssemblyRequest) -> ContextAssemblyResult:
    ctx = _shared(request)
    params = ctx["params"]
    variant = get_planner_prompt(str(_p(params, "plan_planner_prompt")))
    defaults = {
        "retrieval_mode": str(_p(params, "plan_retrieval_mode")),
        "support_policy": str(_p(params, "plan_support_policy")),
        "selection_policy": str(_p(params, "plan_selection_policy")),
        "ordering_policy": str(_p(params, "plan_ordering_policy")),
    }
    planned = plan(
        ctx["question_text"],
        ctx["model"],
        variant,
        JsonKVCache(ctx["cache_path"], section="plan_planner"),
        defaults=defaults,
        max_nodes=int(_p(params, "plan_max_nodes")),
        max_tokens=int(_p(params, "plan_llm_max_tokens")),
    )
    contract = planned["contract"]
    policies = {
        "retrieval_mode": contract.retrieval_mode,
        "support_policy": contract.support_policy,
        "selection_policy": contract.selection_policy,
        "ordering_policy": contract.ordering_policy,
    }
    trace = {
        "method": "plan_dynamic",
        "source": "evidence-planning experiment (LLM-planned procedure)",
        "deviations": ["dynamic procedure: the planner chooses retrieval/selection/ordering"],
        "planner_prompt_variant": planned["prompt_variant"],
        "planner_prompt": planned["prompt"],
        "planner_raw_output": planned["raw_output"],
        "planner_cache_hit": planned["cache_hit"],
        "contract": contract.to_dict(),
    }
    return _run(request, ctx, contract.nodes, policies, trace, "plan_dynamic")
