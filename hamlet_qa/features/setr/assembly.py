"""SetR-lite context assembly adapter."""

from __future__ import annotations

from hamlet_qa.core.context import ContextAssemblyRequest, ContextAssemblyResult
from hamlet_qa.features.setr.selector import select_setr_lite


def assemble_setr_lite(request: ContextAssemblyRequest) -> ContextAssemblyResult:
    if not request.retrieval_trace:
        raise ValueError("setr requires a dense retrieval trace")
    original_hit_chunk_ids = [str(row["chunk_id"]) for row in request.retrieval_trace]
    role_templates = (
        request.domain_kg.evidence_role_templates
        if request.domain_kg is not None
        else None
    )
    assembled = select_setr_lite(
        request.question,
        original_hit_chunk_ids,
        request.chunk_lookup,
        request.context_budget,
        retrieval_trace=request.retrieval_trace,
        role_templates=role_templates,
        cache_path=request.setr_cache_path,
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
