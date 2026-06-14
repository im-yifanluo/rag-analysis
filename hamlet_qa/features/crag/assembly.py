"""CRAG context assembly adapter."""

from __future__ import annotations

from typing import Any

from hamlet_qa.core.context import (
    ContextAssemblyRequest,
    ContextAssemblyResult,
    make_pseudo_chunk,
    truncate_text_to_word_budget,
)
from hamlet_qa.features.crag.corrective import (
    action_from_scores,
    combine_knowledge,
    doc_flags,
    refine_passages,
)
from hamlet_qa.features.crag.rewrite import rewrite_query


CRAG_PSEUDO_CHUNK_ID = "crag_refined_knowledge"

CRAG_DEVIATIONS = [
    "Qwen reranker scores replace the fine-tuned T5-large evaluator "
    "(thresholds recalibrated via cli/calibrate_crag.py)",
    "web search replaced by keyword rewrite + BM25 over all document chunks",
    "GPT-3.5 keyword extractor replaced by the local reader model",
    "refined knowledge word-truncated to the context budget",
]


def assemble_crag(request: ContextAssemblyRequest) -> ContextAssemblyResult:
    if not request.retrieval_trace:
        raise ValueError("crag requires a dense retrieval trace")
    evaluator = request.feature_handles.get("crag_evaluator")
    if evaluator is None:
        raise ValueError(
            "crag requires an evaluator handle "
            "(feature_handles['crag_evaluator'])"
        )

    params = request.feature_params
    ndocs = int(params.get("crag_ndocs", 10))
    upper_threshold = float(params.get("crag_upper_threshold", 2.5))
    lower_threshold = float(params.get("crag_lower_threshold", 0.875))
    decompose_mode = str(params.get("crag_decompose_mode", "excerption"))
    external_top_k = int(params.get("crag_external_top_k", 5))

    question_text = request.question.question
    original_hit_chunk_ids = [str(row["chunk_id"]) for row in request.retrieval_trace]
    top_rows = [
        row
        for row in request.retrieval_trace[:ndocs]
        if str(row["chunk_id"]) in request.chunk_lookup
    ]
    doc_scores: list[float] = []
    for row in top_rows:
        score = row.get("rerank_score")
        if score is None:
            raise ValueError(
                "crag uses reranker scores as evaluator confidence; run with "
                "a reranker model so the dense trace carries rerank_score."
            )
        doc_scores.append(float(score))

    action = action_from_scores(doc_scores, upper_threshold, lower_threshold)
    trace: dict[str, Any] = {
        "method": "crag",
        "source": "third_party/CorrectiveRAG/CRAG scripts port",
        "action": action,
        "evaluated_chunk_ids": [str(row["chunk_id"]) for row in top_rows],
        "evaluator_scores": doc_scores,
        "doc_flags": doc_flags(doc_scores, upper_threshold, lower_threshold),
        "upper_threshold": upper_threshold,
        "lower_threshold": lower_threshold,
        "decompose_mode": decompose_mode,
        "deviations": list(CRAG_DEVIATIONS),
    }

    internal_text = ""
    if action in {"correct", "ambiguous"}:
        internal_passages = [
            str(request.chunk_lookup[str(row["chunk_id"])]["text"]) for row in top_rows
        ]
        internal = refine_passages(
            internal_passages,
            question_text,
            evaluator,
            decompose_mode,
        )
        internal_text = internal["refined_text"]
        trace["internal_refinement"] = {
            "selected_indices": internal["selected_indices"],
            "strip_scores": internal["strip_scores"],
        }

    external_text = ""
    if action in {"incorrect", "ambiguous"}:
        external = external_knowledge(request, question_text, external_top_k, evaluator, decompose_mode)
        external_text = external.pop("refined_text")
        trace["external_knowledge"] = external

    if action == "correct":
        knowledge_text = internal_text
    elif action == "incorrect":
        knowledge_text = external_text
    else:
        knowledge_text = combine_knowledge(internal_text, external_text)

    truncated = truncate_text_to_word_budget(knowledge_text.strip(), request.context_budget)
    trace["refined_knowledge"] = knowledge_text
    trace["truncated_to_budget"] = truncated != knowledge_text.strip()
    if not truncated:
        return ContextAssemblyResult(
            selected_chunk_ids=[],
            selected_chunks=[],
            original_hit_chunk_ids=original_hit_chunk_ids,
            retrieval_trace=[dict(row) for row in request.retrieval_trace],
            retrieval_method=f"crag_{action}_empty",
            prompt_order="crag_refined_knowledge",
            context_assembly_trace=trace,
        )
    pseudo_chunk = make_pseudo_chunk(
        chunk_id=CRAG_PSEUDO_CHUNK_ID,
        text=truncated,
        scene_title="CRAG refined knowledge",
        scene_id="crag",
    )
    return ContextAssemblyResult(
        selected_chunk_ids=[CRAG_PSEUDO_CHUNK_ID],
        selected_chunks=[pseudo_chunk],
        original_hit_chunk_ids=original_hit_chunk_ids,
        retrieval_trace=[dict(row) for row in request.retrieval_trace],
        retrieval_method=f"crag_{action}",
        prompt_order="crag_refined_knowledge",
        context_assembly_trace=trace,
    )


def external_knowledge(
    request: ContextAssemblyRequest,
    question_text: str,
    external_top_k: int,
    evaluator: Any,
    decompose_mode: str,
) -> dict[str, Any]:
    """Corrective re-retrieval over the whole document (web-search substitute)."""
    reretriever = request.feature_handles.get("crag_reretriever")
    if reretriever is None:
        raise ValueError(
            "crag requires a doc-wide re-retriever handle "
            "(feature_handles['crag_reretriever'])"
        )
    if request.selector_model is None:
        raise ValueError("crag query rewriting requires the reader model")
    rewrite = rewrite_query(
        question_text,
        request.selector_model,
        cache_path=request.feature_params.get("crag_rewrite_cache_path"),
    )
    hits = reretriever.retrieve(rewrite["rewritten_query"], external_top_k)
    passages = [
        str(request.chunk_lookup[str(hit["chunk_id"])]["text"])
        for hit in hits
        if str(hit["chunk_id"]) in request.chunk_lookup
    ]
    refined = refine_passages(passages, question_text, evaluator, decompose_mode)
    return {
        "refined_text": refined["refined_text"],
        "rewritten_query": rewrite["rewritten_query"],
        "rewrite_raw_output": rewrite.get("raw_output"),
        "rewrite_cache_hit": rewrite.get("cache_hit"),
        "reretrieved_chunk_ids": [str(hit["chunk_id"]) for hit in hits],
        "reretrieval_scores": [float(hit.get("score", 0.0)) for hit in hits],
        "selected_indices": refined["selected_indices"],
        "strip_scores": refined["strip_scores"],
    }
