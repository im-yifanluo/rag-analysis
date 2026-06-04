"""Context ordering treatments over the same relevance-retrieved hits."""

from __future__ import annotations

from typing import Literal

from hamlet_qa.core.context import (
    ContextAssemblyRequest,
    ContextAssemblyResult,
    select_chunk_ids_for_budget,
    selected_chunks,
    sort_by_document_order,
    stable_random_order,
)


def assemble_dense_document_order(request: ContextAssemblyRequest) -> ContextAssemblyResult:
    return assemble_dense_ordering(
        request,
        order_mode="document",
        prompt_order="dense_hits_document_order",
    )


def assemble_dense_random_order(request: ContextAssemblyRequest) -> ContextAssemblyResult:
    return assemble_dense_ordering(
        request,
        order_mode="random",
        prompt_order="dense_hits_random_order",
    )


def assemble_dense_ordering(
    request: ContextAssemblyRequest,
    order_mode: Literal["document", "random"],
    prompt_order: str,
) -> ContextAssemblyResult:
    if not request.retrieval_trace:
        raise ValueError(f"{request.treatment} requires a dense retrieval trace")
    original_hit_chunk_ids = [str(row["chunk_id"]) for row in request.retrieval_trace]
    selected_ids = select_chunk_ids_for_budget(
        original_hit_chunk_ids,
        request.chunk_lookup,
        request.context_budget,
    )
    if order_mode == "document":
        selected_ids = sort_by_document_order(selected_ids, request.chunk_lookup)
    elif order_mode == "random":
        selected_ids = stable_random_order(
            selected_ids,
            request.random_seed,
            f"{request.question.id}:{request.context_budget}:{request.treatment}",
        )

    retrieval_method = str(
        request.retrieval_trace[0].get("retrieval_method", "dense_faiss_reranked")
    )
    return ContextAssemblyResult(
        selected_chunk_ids=selected_ids,
        selected_chunks=selected_chunks(selected_ids, request.chunk_lookup),
        original_hit_chunk_ids=original_hit_chunk_ids,
        retrieval_trace=[dict(row) for row in request.retrieval_trace],
        retrieval_method=retrieval_method,
        prompt_order=prompt_order,
    )
