"""Treatment registry for context assembly features."""

from __future__ import annotations

from hamlet_qa.core.context import TreatmentSpec
from hamlet_qa.features.baseline.assembly import (
    assemble_closed_book,
    assemble_dense_rank_order,
    assemble_gold_evidence,
    assemble_sparse_bm25,
)
from hamlet_qa.features.domain.assembly import assemble_domain_kg_lite
from hamlet_qa.features.ordering.assembly import (
    assemble_dense_document_order,
    assemble_dense_random_order,
)
from hamlet_qa.features.setr.assembly import assemble_setr_lite


TREATMENT_REGISTRY: dict[str, TreatmentSpec] = {
    "closed_book": TreatmentSpec("closed_book", assemble_closed_book),
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
        assemble_setr_lite,
        retrieval_source="dense",
        uses_domain_kg=True,
    ),
    "domain": TreatmentSpec(
        "domain",
        assemble_domain_kg_lite,
        retrieval_source="dense",
        uses_domain_kg=True,
    ),
}

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


def known_treatment_names() -> set[str]:
    return set(TREATMENT_REGISTRY)
