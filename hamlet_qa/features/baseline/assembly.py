"""Baseline relevance and control context assembly treatments."""

from __future__ import annotations

from hamlet_qa.core.context import (
    ContextAssemblyRequest,
    ContextAssemblyResult,
    select_chunk_ids_for_budget,
    selected_chunks,
    sort_by_document_order,
)


def assemble_closed_book(request: ContextAssemblyRequest) -> ContextAssemblyResult:
    del request
    return ContextAssemblyResult(
        selected_chunk_ids=[],
        selected_chunks=[],
        prompt_order="none",
    )


def assemble_gold_evidence(request: ContextAssemblyRequest) -> ContextAssemblyResult:
    original_hit_chunk_ids = sort_by_document_order(
        list(request.question.derived_gold_chunk_ids),
        request.chunk_lookup,
    )
    selected_ids = select_chunk_ids_for_budget(
        original_hit_chunk_ids,
        request.chunk_lookup,
        request.context_budget,
    )
    return ContextAssemblyResult(
        selected_chunk_ids=selected_ids,
        selected_chunks=selected_chunks(selected_ids, request.chunk_lookup),
        original_hit_chunk_ids=original_hit_chunk_ids,
        retrieval_method="gold",
        prompt_order="gold_chunks_document_order",
    )


def assemble_dense_rank_order(request: ContextAssemblyRequest) -> ContextAssemblyResult:
    if not request.retrieval_trace:
        raise ValueError(f"{request.treatment} requires a dense retrieval trace")
    original_hit_chunk_ids = [str(row["chunk_id"]) for row in request.retrieval_trace]
    selected_ids = select_chunk_ids_for_budget(
        original_hit_chunk_ids,
        request.chunk_lookup,
        request.context_budget,
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
        prompt_order="dense_reranker_rank",
    )


def assemble_sparse_bm25(request: ContextAssemblyRequest) -> ContextAssemblyResult:
    if not request.retrieval_trace:
        raise ValueError("sparse_bm25 requires a sparse retrieval trace")
    original_hit_chunk_ids = [str(row["chunk_id"]) for row in request.retrieval_trace]
    selected_ids = select_chunk_ids_for_budget(
        original_hit_chunk_ids,
        request.chunk_lookup,
        request.context_budget,
    )
    retrieval_method = str(request.retrieval_trace[0].get("retrieval_method", "bm25"))
    return ContextAssemblyResult(
        selected_chunk_ids=selected_ids,
        selected_chunks=selected_chunks(selected_ids, request.chunk_lookup),
        original_hit_chunk_ids=original_hit_chunk_ids,
        retrieval_trace=[dict(row) for row in request.retrieval_trace],
        retrieval_method=retrieval_method,
        prompt_order="bm25_score",
    )
