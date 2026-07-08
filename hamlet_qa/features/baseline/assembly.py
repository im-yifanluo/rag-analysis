"""Baseline relevance and control context assembly treatments."""

from __future__ import annotations

from typing import Any

from hamlet_qa.core.context import (
    ContextAssemblyRequest,
    ContextAssemblyResult,
    make_pseudo_chunk,
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


def assemble_full_document(request: ContextAssemblyRequest) -> ContextAssemblyResult:
    """Long-context control: the entire play as ONE clean block.

    The "reader has everything" upper bound — no retrieval, no selection, no
    per-chunk headers, no overlap double-counting. Deliberately bypasses
    `select_chunk_ids_for_budget`; the token budget does not apply. The raw play
    text is threaded in as the ``full_document_text`` handle by ``run_experiment``.
    Note: ``make_pseudo_chunk`` records a whitespace-word ``token_count``, so the
    row's ``context_tokens`` under-reads the real prompt; ``prompt_tokens`` (real
    tokenizer) is the meaningful size (~68-80K here).
    """
    text = str(request.feature_handles.get("full_document_text") or "")
    if not text:
        raise ValueError(
            "full_document requires feature_handles['full_document_text'] "
            "(the raw document text)"
        )
    chunk = make_pseudo_chunk(
        "hamlet_full_text",
        text,
        scene_title="The Tragedy of Hamlet (full text)",
        scene_id="full_document",
    )
    return ContextAssemblyResult(
        selected_chunk_ids=[chunk["chunk_id"]],
        selected_chunks=[chunk],
        original_hit_chunk_ids=[chunk["chunk_id"]],
        retrieval_method="full_document",
        prompt_order="document_order",
    )


def minimal_quote_cover_chunk_ids(
    question: Any,
    chunk_lookup: dict[str, Any],
) -> list[str]:
    """Pick one matched chunk per required quote, preferring reuse.

    Overlapping chunks can satisfy the same quote; a greedy budget fill over
    all gold chunks can then exhaust the budget on redundant chunks and drop
    the only chunk holding a later quote. Covering each quote first keeps
    quote recall at 1.0 whenever a covering set fits the budget.
    """
    cover: list[str] = []
    for evidence_quote in question.required_evidence_quotes:
        matched = [
            chunk_id
            for chunk_id in evidence_quote.matched_chunk_ids
            if chunk_id in chunk_lookup
        ]
        if not matched:
            continue
        if any(chunk_id in cover for chunk_id in matched):
            continue
        cover.append(matched[0])
    return cover


def assemble_gold_evidence(request: ContextAssemblyRequest) -> ContextAssemblyResult:
    original_hit_chunk_ids = sort_by_document_order(
        list(request.question.derived_gold_chunk_ids),
        request.chunk_lookup,
    )
    quote_cover_ids = sort_by_document_order(
        minimal_quote_cover_chunk_ids(request.question, request.chunk_lookup),
        request.chunk_lookup,
    )
    remaining_gold_ids = [
        chunk_id
        for chunk_id in original_hit_chunk_ids
        if chunk_id not in quote_cover_ids
    ]
    selected_ids = select_chunk_ids_for_budget(
        quote_cover_ids + remaining_gold_ids,
        request.chunk_lookup,
        request.context_budget,
    )
    selected_ids = sort_by_document_order(selected_ids, request.chunk_lookup)
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
