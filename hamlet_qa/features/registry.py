"""Treatment registry for context assembly features."""

from __future__ import annotations

from hamlet_qa.core.context import TreatmentSpec
from hamlet_qa.features.crag.assembly import assemble_crag
from hamlet_qa.features.baseline.assembly import (
    assemble_closed_book,
    assemble_dense_rank_order,
    assemble_full_document,
    assemble_gold_evidence,
    assemble_sparse_bm25,
)
from hamlet_qa.features.domain.assembly import assemble_domain_kg_lite
from hamlet_qa.features.evidence_plan.assembly import (
    assemble_plan_dynamic,
    assemble_plan_fixed,
)
from hamlet_qa.features.macrag.assembly import assemble_macrag
from hamlet_qa.features.ordering.assembly import (
    assemble_dense_document_order,
    assemble_dense_random_order,
)
from hamlet_qa.features.reader_support.assembly import assemble_reader_support
from hamlet_qa.features.recomp.assembly import (
    assemble_recomp_abstractive,
    assemble_recomp_extractive,
)
from hamlet_qa.features.setr.assembly import assemble_setr


TREATMENT_REGISTRY: dict[str, TreatmentSpec] = {
    "closed_book": TreatmentSpec("closed_book", assemble_closed_book),
    "full_document": TreatmentSpec("full_document", assemble_full_document),
    "gold_evidence": TreatmentSpec("gold_evidence", assemble_gold_evidence),
    "dense_reranked": TreatmentSpec(
        "dense_reranked",
        assemble_dense_rank_order,
        retrieval_source="dense",
    ),
    "dense_document_order": TreatmentSpec(
        "dense_document_order",
        assemble_dense_document_order,
        retrieval_source="dense",
    ),
    "dense_random_order": TreatmentSpec(
        "dense_random_order",
        assemble_dense_random_order,
        retrieval_source="dense",
    ),
    "sparse_bm25": TreatmentSpec(
        "sparse_bm25",
        assemble_sparse_bm25,
        retrieval_source="sparse",
    ),
    "setr": TreatmentSpec(
        "setr",
        assemble_setr,
        retrieval_source="dense",
        uses_llm_assembly=True,
    ),
    "domain": TreatmentSpec(
        "domain",
        assemble_domain_kg_lite,
        retrieval_source="dense",
        uses_domain_kg=True,
    ),
    "recomp_extractive": TreatmentSpec(
        "recomp_extractive",
        assemble_recomp_extractive,
        retrieval_source="dense",
    ),
    "recomp_abstractive": TreatmentSpec(
        "recomp_abstractive",
        assemble_recomp_abstractive,
        retrieval_source="dense",
    ),
    "crag": TreatmentSpec(
        "crag",
        assemble_crag,
        retrieval_source="dense",
        uses_llm_assembly=True,
    ),
    "macrag": TreatmentSpec(
        "macrag",
        assemble_macrag,
        retrieval_source="macrag",
    ),
    "reader_support": TreatmentSpec(
        "reader_support",
        assemble_reader_support,
        retrieval_source="dense",
        uses_llm_assembly=True,
    ),
    "plan_fixed": TreatmentSpec(
        "plan_fixed",
        assemble_plan_fixed,
        retrieval_source="dense",
        uses_llm_assembly=True,
        needs_node_retriever=True,
    ),
    "plan_dynamic": TreatmentSpec(
        "plan_dynamic",
        assemble_plan_dynamic,
        retrieval_source="dense",
        uses_llm_assembly=True,
        needs_node_retriever=True,
    ),
}


# ---------------------------------------------------------------------------
# Evidence-planning experiment arms.
#
# The evidence-planning study is ONE experiment: it varies the decomposition /
# planner prompt (and, for the fixed pipeline, the retrieval mode) while holding
# the rest of the pipeline constant. Each configuration is registered as its own
# treatment so the whole matrix runs under a SINGLE model load and lands in ONE
# results.jsonl — every row tagged by its arm, comparable via the existing
# per-treatment reporting. An arm inherits everything from its base treatment
# and only overlays `feature_params` through `param_overrides`; the assembly
# function is unchanged, so `context_assembly_trace["method"]` stays the base
# name and the plan_eval metric keeps recognising these rows.
# ---------------------------------------------------------------------------

# (arm name, base treatment, feature_params overrides)
_PLAN_ARMS: list[tuple[str, str, dict[str, object]]] = [
    # Fixed pipeline: 3 decomposition prompts x 2 retrieval modes.
    ("plan_fixed_subq_par", "plan_fixed",
     {"plan_decomp_prompt": "split_questions", "plan_retrieval_mode": "parallel"}),
    ("plan_fixed_subq_seq", "plan_fixed",
     {"plan_decomp_prompt": "split_questions", "plan_retrieval_mode": "sequential"}),
    ("plan_fixed_inforeq_par", "plan_fixed",
     {"plan_decomp_prompt": "list_requirements", "plan_retrieval_mode": "parallel"}),
    ("plan_fixed_inforeq_seq", "plan_fixed",
     {"plan_decomp_prompt": "list_requirements", "plan_retrieval_mode": "sequential"}),
    ("plan_fixed_strategy_par", "plan_fixed",
     {"plan_decomp_prompt": "reason_then_plan", "plan_retrieval_mode": "parallel"}),
    ("plan_fixed_strategy_seq", "plan_fixed",
     {"plan_decomp_prompt": "reason_then_plan", "plan_retrieval_mode": "sequential"}),
    # Dynamic pipeline: 2 planner prompts (the planner picks the retrieval mode).
    ("plan_dynamic_contract", "plan_dynamic",
     {"plan_planner_prompt": "contract_v1"}),
    ("plan_dynamic_strategy", "plan_dynamic",
     {"plan_planner_prompt": "strategy_contract"}),
]


def _arm_from_base(name: str, base_name: str, overrides: dict[str, object]) -> TreatmentSpec:
    base = TREATMENT_REGISTRY[base_name]
    return TreatmentSpec(
        name=name,
        assemble=base.assemble,
        retrieval_source=base.retrieval_source,
        uses_domain_kg=base.uses_domain_kg,
        uses_llm_assembly=base.uses_llm_assembly,
        needs_node_retriever=base.needs_node_retriever,
        base_treatment=base_name,
        param_overrides=dict(overrides),
    )


for _arm_name, _arm_base, _arm_overrides in _PLAN_ARMS:
    TREATMENT_REGISTRY[_arm_name] = _arm_from_base(_arm_name, _arm_base, _arm_overrides)

# Names of the evidence-planning sweep arms, in matrix order (for run recipes).
PLAN_SWEEP_ARMS: list[str] = [name for name, _, _ in _PLAN_ARMS]

DENSE_TREATMENTS = {
    name
    for name, treatment in TREATMENT_REGISTRY.items()
    if treatment.retrieval_source == "dense"
}
SPARSE_TREATMENTS = {
    name
    for name, treatment in TREATMENT_REGISTRY.items()
    if treatment.retrieval_source == "sparse"
}


def get_treatment(treatment: str) -> TreatmentSpec:
    try:
        return TREATMENT_REGISTRY[treatment]
    except KeyError as error:
        raise ValueError(f"Unknown treatment: {treatment}") from error


def treatments_using_domain_kg(treatments: list[str]) -> bool:
    return any(get_treatment(treatment).uses_domain_kg for treatment in treatments)


def treatments_using_llm_assembly(treatments: list[str]) -> bool:
    return any(get_treatment(treatment).uses_llm_assembly for treatment in treatments)


def known_treatment_names() -> set[str]:
    return set(TREATMENT_REGISTRY)
