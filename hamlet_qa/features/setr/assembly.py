"""SetR context assembly adapter."""

from __future__ import annotations

from hamlet_qa.core.context import ContextAssemblyRequest, ContextAssemblyResult
from hamlet_qa.features.setr.selector import SetRSelectionError, select_setr


def assemble_setr(request: ContextAssemblyRequest) -> ContextAssemblyResult:
    if not request.retrieval_trace:
        raise ValueError("setr requires a dense retrieval trace")
    if request.selector_model is None:
        raise SetRSelectionError("setr requires a selector model")

    original_hit_chunk_ids = [str(row["chunk_id"]) for row in request.retrieval_trace]
    assembled = select_setr(
        request.question,
        original_hit_chunk_ids,
        request.chunk_lookup,
        request.context_budget,
        selector_model=request.selector_model,
        retrieval_trace=request.retrieval_trace,
        cache_path=request.setr_cache_path,
        selector_max_tokens=request.setr_selector_max_tokens,
        max_passages=request.setr_max_passages,
    )
    return ContextAssemblyResult(
        selected_chunk_ids=list(assembled["selected_chunk_ids"]),
        selected_chunks=[dict(chunk) for chunk in assembled["selected_chunks"]],
        original_hit_chunk_ids=original_hit_chunk_ids,
        retrieval_trace=[dict(row) for row in request.retrieval_trace],
        retrieval_method=str(assembled["retrieval_method"]),
        prompt_order=str(assembled["prompt_order"]),
        context_assembly_trace=dict(assembled["context_assembly_trace"]),
    )
