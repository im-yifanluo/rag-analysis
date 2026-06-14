"""RECOMP context assembly adapters (extractive and abstractive)."""

from __future__ import annotations

from typing import Any

from hamlet_qa.core.context import (
    ContextAssemblyRequest,
    ContextAssemblyResult,
    make_pseudo_chunk,
    truncate_text_to_word_budget,
)
from hamlet_qa.core.llm_cache import JsonKVCache, stable_hash
from hamlet_qa.features.recomp.compressor import (
    RECOMP_PROMPTED_ABSTRACTIVE_PROMPT,
    build_prompted_abstractive_prompt,
)


RECOMP_DEVIATIONS = [
    "HotpotQA-trained compressor checkpoints used zero-shot on Shakespeare",
    "harness 256-token chunks instead of 100-word Wikipedia passages",
    "harness prompt builder instead of the paper's few-shot QA prompt",
    "summary word-truncated to the context budget",
]


def recomp_input_chunks(request: ContextAssemblyRequest) -> list[dict[str, Any]]:
    if not request.retrieval_trace:
        raise ValueError(f"{request.treatment} requires a dense retrieval trace")
    input_docs = int(request.feature_params.get("recomp_input_docs", 5))
    chunk_ids = [
        str(row["chunk_id"])
        for row in request.retrieval_trace[:input_docs]
        if str(row["chunk_id"]) in request.chunk_lookup
    ]
    return [dict(request.chunk_lookup[chunk_id]) for chunk_id in chunk_ids]


def summary_assembly_result(
    request: ContextAssemblyRequest,
    summary: str,
    pseudo_chunk_id: str,
    scene_title: str,
    trace: dict[str, Any],
) -> ContextAssemblyResult:
    original_hit_chunk_ids = [str(row["chunk_id"]) for row in request.retrieval_trace]
    truncated = truncate_text_to_word_budget(summary.strip(), request.context_budget)
    trace = dict(trace)
    trace["summary"] = summary
    trace["summary_truncated_to_budget"] = truncated != summary.strip()
    trace["deviations"] = list(RECOMP_DEVIATIONS)
    if not truncated:
        # Selective augmentation: an empty summary means "retrieval does not
        # help"; the reader answers closed-book.
        trace["empty_summary"] = True
        return ContextAssemblyResult(
            selected_chunk_ids=[],
            selected_chunks=[],
            original_hit_chunk_ids=original_hit_chunk_ids,
            retrieval_trace=[dict(row) for row in request.retrieval_trace],
            retrieval_method=f"{request.treatment}_empty_summary",
            prompt_order="recomp_summary",
            context_assembly_trace=trace,
        )
    pseudo_chunk = make_pseudo_chunk(
        chunk_id=pseudo_chunk_id,
        text=truncated,
        scene_title=scene_title,
        scene_id="recomp_summary",
    )
    return ContextAssemblyResult(
        selected_chunk_ids=[pseudo_chunk_id],
        selected_chunks=[pseudo_chunk],
        original_hit_chunk_ids=original_hit_chunk_ids,
        retrieval_trace=[dict(row) for row in request.retrieval_trace],
        retrieval_method=request.treatment,
        prompt_order="recomp_summary",
        context_assembly_trace=trace,
    )


def precomputed_summary(
    request: ContextAssemblyRequest,
) -> dict[str, Any]:
    summaries = request.feature_handles.get("recomp_summaries")
    if not isinstance(summaries, dict):
        raise ValueError(
            f"{request.treatment} requires precomputed compressor summaries; "
            "run_experiment computes them before the reader loads "
            "(feature_handles['recomp_summaries'])."
        )
    key = f"{request.treatment}:{request.question.id}"
    record = summaries.get(key)
    if not isinstance(record, dict):
        raise ValueError(f"Missing precomputed RECOMP summary for {key}")
    return dict(record)


def assemble_recomp_extractive(request: ContextAssemblyRequest) -> ContextAssemblyResult:
    input_chunks = recomp_input_chunks(request)
    record = precomputed_summary(request)
    trace = {
        "method": "recomp_extractive",
        "source": "third_party/RECOMP run_extractive_compressor.py port",
        "input_chunk_ids": [chunk["chunk_id"] for chunk in input_chunks],
        "compressor_model": record.get("compressor_model"),
        "selected_sentences": record.get("selected_sentences", []),
        "num_input_sentences": record.get("num_input_sentences"),
    }
    return summary_assembly_result(
        request,
        str(record.get("summary", "")),
        pseudo_chunk_id="recomp_extractive_summary",
        scene_title="RECOMP extractive summary",
        trace=trace,
    )


def assemble_recomp_abstractive(request: ContextAssemblyRequest) -> ContextAssemblyResult:
    input_chunks = recomp_input_chunks(request)
    mode = str(request.feature_params.get("recomp_abstractive_mode", "t5"))
    if mode == "prompted_qwen":
        record = prompted_qwen_summary(request, input_chunks)
    else:
        record = precomputed_summary(request)
    trace = {
        "method": "recomp_abstractive",
        "source": "third_party/RECOMP abstractive compressor",
        "mode": mode,
        "input_chunk_ids": [chunk["chunk_id"] for chunk in input_chunks],
        "compressor_model": record.get("compressor_model"),
        "compressor_input": record.get("compressor_input"),
        "cache_hit": record.get("cache_hit"),
    }
    return summary_assembly_result(
        request,
        str(record.get("summary", "")),
        pseudo_chunk_id="recomp_abstractive_summary",
        scene_title="RECOMP abstractive summary",
        trace=trace,
    )


def prompted_qwen_summary(
    request: ContextAssemblyRequest,
    input_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Paper Table 8 prompt against the resident reader model."""
    if request.selector_model is None:
        raise ValueError(
            "recomp_abstractive in prompted_qwen mode requires the reader model"
        )
    prompt = build_prompted_abstractive_prompt(request.question.question, input_chunks)
    model_name = str(getattr(request.selector_model, "model_name", "reader_model"))
    cache_path = request.feature_params.get("recomp_cache_path")
    cache = JsonKVCache(cache_path, section="recomp_prompted_abstractive")
    cache_key = stable_hash({"prompt": prompt, "model": model_name})
    cached = cache.get(cache_key)
    if cached is not None:
        return dict(cached, cache_hit=True)
    summary = request.selector_model.generate(
        "You are a helpful assistant.",
        prompt,
    )
    record = {
        "summary": summary,
        "compressor_input": prompt,
        "compressor_model": model_name,
        "prompt_template": RECOMP_PROMPTED_ABSTRACTIVE_PROMPT,
    }
    cache.set(cache_key, record)
    cache.save()
    return dict(record, cache_hit=False)
