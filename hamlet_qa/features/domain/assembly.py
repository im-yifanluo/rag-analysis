"""Domain-KG-lite context assembly adapter."""

from __future__ import annotations

from hamlet_qa.core.context import ContextAssemblyRequest, ContextAssemblyResult
from hamlet_qa.features.domain.kg import select_domain_kg_lite


def assemble_domain_kg_lite(request: ContextAssemblyRequest) -> ContextAssemblyResult:
    if not request.retrieval_trace:
        raise ValueError("domain requires a dense retrieval trace")
    if request.domain_kg is None:
        raise ValueError("domain requires a domain knowledge graph")
    original_hit_chunk_ids = [str(row["chunk_id"]) for row in request.retrieval_trace]
    assembled = select_domain_kg_lite(
        request.question,
        original_hit_chunk_ids,
        request.chunk_lookup,
        request.context_budget,
        retrieval_trace=request.retrieval_trace,
        domain_kg=request.domain_kg,
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
