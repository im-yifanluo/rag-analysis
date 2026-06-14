"""Experiment orchestration, treatment selection, and result logging."""

from __future__ import annotations

import gc
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from hamlet_qa.features.domain.kg import DomainKnowledgeGraph
from hamlet_qa.features.registry import (
    DENSE_TREATMENTS,
    SPARSE_TREATMENTS,
    get_treatment,
    known_treatment_names,
    treatments_using_domain_kg,
    treatments_using_llm_assembly,
)
from hamlet_qa.core.config import RunConfig
from hamlet_qa.core.io import append_jsonl, dump_json, load_jsonl
from hamlet_qa.core.questions import (
    Question,
    load_questions,
    normalize_text,
    validate_questions,
)
from hamlet_qa.core.generation import VLLMReader
from hamlet_qa.core.context import (
    ContextAssemblyRequest,
    ContextAssemblyResult,
    chunks_by_id,
    document_order_chunk_ids,
)
from hamlet_qa.core.prompts import (
    HamletQAPromptBuilder,
    PromptBundle,
    TokenizerPromptFormatter,
)
from hamlet_qa.core.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    SentenceTransformerEmbedder,
)


class ReaderLike(Protocol):
    model_max_context: int | None

    def format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        ...

    def count_tokens(self, text: str) -> int:
        ...

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        ...


class RetrieverLike(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        ...


def clear_cuda_cache() -> None:
    """Release PyTorch CUDA cache after unloading retrieval models."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except RuntimeError:
        pass


def required_quotes_present_in_context(
    question: Question,
    selected_chunk_ids: list[str],
    selected_chunks: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Check quote presence by matched chunk ID or by selected context text.

    The text check matters for treatments whose context contains synthetic
    pseudo-chunks (CRAG refined knowledge, RECOMP summaries, MacRAG merged
    blocks): evidence can be present verbatim without any gold chunk ID being
    selected.
    """
    selected = set(selected_chunk_ids)
    context_text = ""
    if selected_chunks:
        context_text = normalize_text(
            "\n".join(str(chunk.get("text", "")) for chunk in selected_chunks)
        )
    present: list[dict[str, Any]] = []
    for index, evidence_quote in enumerate(question.required_evidence_quotes):
        matched = list(evidence_quote.matched_chunk_ids)
        present_by_chunk_id = bool(selected & set(matched))
        present_by_text = bool(
            context_text and normalize_text(evidence_quote.quote) in context_text
        )
        present.append(
            {
                "quote_index": index,
                "quote": evidence_quote.quote,
                "role": evidence_quote.role,
                "matched_chunk_ids": matched,
                "present": present_by_chunk_id or present_by_text,
                "present_by_chunk_id": present_by_chunk_id,
                "present_by_text": present_by_text,
            }
        )
    return present


def evidence_chunk_recall(selected_chunk_ids: list[str], gold_chunk_ids: list[str]) -> float | None:
    if not gold_chunk_ids:
        return None
    return len(set(selected_chunk_ids) & set(gold_chunk_ids)) / len(set(gold_chunk_ids))


def evidence_quote_recall(quote_presence: list[dict[str, Any]]) -> float | None:
    if not quote_presence:
        return None
    return sum(1 for row in quote_presence if row["present"]) / len(quote_presence)


def retrieval_scores_for(
    selected_chunk_ids: list[str],
    retrieval_trace: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if retrieval_trace is None:
        return []
    score_by_id = {
        str(row["chunk_id"]): dict(
            {
                "chunk_id": str(row["chunk_id"]),
                "rank": int(row["rank"]),
                "score": row.get("score"),
            },
            **{
                key: row[key]
                for key in (
                    "dense_rank",
                    "dense_score",
                    "rerank_score",
                    "sparse_rank",
                    "sparse_score",
                    "retrieval_method",
                )
                if key in row
            },
        )
        for row in retrieval_trace
    }
    return [
        score_by_id[chunk_id]
        for chunk_id in selected_chunk_ids
        if chunk_id in score_by_id
    ]


def prepare_treatment(
    question: Question,
    treatment: str,
    context_budget: int,
    chunk_lookup: dict[str, dict[str, Any]],
    doc_order_ids: list[str],
    retrieval_trace: list[dict[str, Any]] | None = None,
    random_seed: int = 13,
    domain_kg: DomainKnowledgeGraph | None = None,
    selector_model: Any = None,
    setr_cache_path: str | Path | None = None,
    setr_max_passages: int = 50,
    setr_selector_max_tokens: int = 4096,
    feature_params: dict[str, Any] | None = None,
    feature_handles: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trace = [dict(row) for row in retrieval_trace] if retrieval_trace else []
    request = ContextAssemblyRequest(
        question=question,
        treatment=treatment,
        context_budget=context_budget,
        chunk_lookup=chunk_lookup,
        doc_order_ids=doc_order_ids,
        retrieval_trace=trace,
        random_seed=random_seed,
        domain_kg=domain_kg,
        selector_model=selector_model,
        setr_cache_path=Path(setr_cache_path) if setr_cache_path else None,
        setr_max_passages=setr_max_passages,
        setr_selector_max_tokens=setr_selector_max_tokens,
        feature_params=dict(feature_params or {}),
        feature_handles=dict(feature_handles or {}),
    )
    assembly = get_treatment(treatment).assemble(request)
    return prepared_context_from_assembly(question, context_budget, assembly)


def prepared_context_from_assembly(
    question: Question,
    context_budget: int,
    assembly: ContextAssemblyResult,
) -> dict[str, Any]:
    selected_ids = assembly.selected_chunk_ids
    final_chunks = assembly.selected_chunks
    quote_presence = required_quotes_present_in_context(
        question,
        selected_ids,
        selected_chunks=final_chunks,
    )
    quote_recall = evidence_quote_recall(quote_presence)
    return {
        "selected_chunk_ids": selected_ids,
        "final_selected_chunk_ids": selected_ids,
        "selected_chunks": final_chunks,
        "context_tokens": assembly.context_tokens,
        "retrieval_trace": assembly.retrieval_trace,
        "retrieval_scores": retrieval_scores_for(selected_ids, assembly.retrieval_trace),
        "retrieval_method": assembly.retrieval_method,
        "prompt_order": assembly.prompt_order,
        "context_assembly_trace": assembly.context_assembly_trace,
        "original_hit_chunk_ids": assembly.original_hit_chunk_ids,
        "evidence_chunk_recall": evidence_chunk_recall(
            selected_ids,
            question.derived_gold_chunk_ids,
        ),
        "evidence_quote_recall": quote_recall,
        "required_quotes_present_in_context": quote_presence,
        "coverage_ratio": (
            assembly.context_tokens / context_budget
            if context_budget > 0
            else None
        ),
    }


def build_result_row(
    question: Question,
    treatment: str,
    context_budget: int,
    prepared: dict[str, Any],
    reader: Any,
    config: RunConfig,
    model_output: str | None,
    prompt_bundle: PromptBundle | None = None,
) -> dict[str, Any]:
    if prompt_bundle is None:
        prompt_bundle = HamletQAPromptBuilder().build(
            question.question,
            prepared["selected_chunks"],
            treatment,
            reader,
        )
    return {
        "run_name": config.run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "question_id": question.id,
        "question": question.question,
        "expected_answer": question.expected_answer,
        "evidence_scope": question.evidence_scope,
        "reasoning_skill": question.reasoning_skill,
        "required_evidence_quotes": [
            quote.to_dict(include_matches=True)
            for quote in question.required_evidence_quotes
        ],
        "derived_gold_chunk_ids": list(question.derived_gold_chunk_ids),
        "treatment": treatment,
        "context_budget": context_budget,
        "selected_chunk_ids": prepared["selected_chunk_ids"],
        "final_selected_chunk_ids": prepared["final_selected_chunk_ids"],
        "original_hit_chunk_ids": prepared["original_hit_chunk_ids"],
        "raw_chunks": prepared["selected_chunks"],
        "evidence_chunk_recall": prepared["evidence_chunk_recall"],
        "evidence_quote_recall": prepared["evidence_quote_recall"],
        "required_quotes_present_in_context": prepared[
            "required_quotes_present_in_context"
        ],
        "context_tokens": prepared["context_tokens"],
        "prompt_tokens": prompt_bundle.prompt_tokens,
        "total_input_tokens": prompt_bundle.prompt_tokens,
        "max_new_tokens": config.max_new_tokens,
        "model_name": config.reader_model,
        "model_max_context": getattr(reader, "model_max_context", None),
        "coverage_ratio": prepared["coverage_ratio"],
        "retrieval_method": prepared["retrieval_method"],
        "retrieval_scores": prepared["retrieval_scores"],
        "retrieval_trace": prepared["retrieval_trace"],
        "prompt_order": prepared["prompt_order"],
        "context_assembly_trace": prepared["context_assembly_trace"],
        "system_prompt": prompt_bundle.system_prompt,
        "user_prompt": prompt_bundle.user_prompt,
        "full_prompt": prompt_bundle.full_prompt,
        "model_output": model_output,
        "failure_label": None,
        "embedding_model": config.embedding_model,
        "reranker_model": config.reranker_model,
        "bm25_k1": config.bm25_k1,
        "bm25_b": config.bm25_b,
        "random_seed": config.random_seed,
        "gpu_layout": config.gpu_layout,
        "embedding_device": config.embedding_device,
        "reranker_device": config.reranker_device,
        "reader_device": config.reader_device,
        "temperature": config.temperature,
        "run_config": config.to_dict(),
    }


def make_reader(config: RunConfig, force_generation_model: bool = False) -> Any:
    if config.prepare_only and not force_generation_model:
        return TokenizerPromptFormatter(config.reader_model)
    return VLLMReader(
        config.reader_model,
        temperature=config.temperature,
        max_new_tokens=config.max_new_tokens,
        tensor_parallel_size=config.tensor_parallel_size,
        gpu_memory_utilization=config.gpu_memory_utilization,
        device=config.reader_device,
    )


def make_dense_retriever(
    config: RunConfig,
    chunks: list[dict[str, Any]],
    include_reranker: bool = True,
) -> DenseRetriever:
    embedder = SentenceTransformerEmbedder(
        config.embedding_model,
        device=config.embedding_device,
        batch_size=config.embedding_batch_size,
    )
    reranker = None
    if include_reranker and config.reranker_model:
        reranker = CrossEncoderReranker(
            config.reranker_model,
            device=config.reranker_device,
            batch_size=config.reranker_batch_size,
        )
    return DenseRetriever(embedder, chunks, reranker=reranker)


def make_reranker(config: RunConfig) -> CrossEncoderReranker:
    if not config.reranker_model:
        raise ValueError("No reranker model configured")
    return CrossEncoderReranker(
        config.reranker_model,
        device=config.reranker_device,
        batch_size=config.reranker_batch_size,
    )


def make_sparse_retriever(config: RunConfig, chunks: list[dict[str, Any]]) -> BM25Retriever:
    return BM25Retriever(chunks, k1=config.bm25_k1, b=config.bm25_b)


def rerank_dense_trace(
    query: str,
    dense_trace: list[dict[str, Any]],
    chunk_lookup: dict[str, dict[str, Any]],
    reranker: CrossEncoderReranker,
) -> list[dict[str, Any]]:
    documents = [str(chunk_lookup[str(row["chunk_id"])]["text"]) for row in dense_trace]
    rerank_scores = reranker.score(query, documents)
    candidates: list[dict[str, Any]] = []
    for row, rerank_score in zip(dense_trace, rerank_scores):
        candidate = dict(row)
        candidate["rerank_score"] = rerank_score
        candidate["score"] = rerank_score
        candidate["retrieval_method"] = "dense_faiss_reranked"
        candidates.append(candidate)
    reranked = sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate["rerank_score"]),
            int(candidate["dense_rank"]),
        ),
    )
    for rank, candidate in enumerate(reranked, start=1):
        candidate["rank"] = rank
    return reranked


def build_retrieval_traces(
    config: RunConfig,
    chunks: list[dict[str, Any]],
    questions: list[Question],
    dense_retriever: RetrieverLike | None = None,
    sparse_retriever: RetrieverLike | None = None,
    feature_handles: dict[str, Any] | None = None,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    traces: dict[str, dict[str, list[dict[str, Any]]]] = {
        question.id: {} for question in questions
    }
    needs_dense_trace = any(treatment in DENSE_TREATMENTS for treatment in config.treatments)
    needs_sparse_trace = any(treatment in SPARSE_TREATMENTS for treatment in config.treatments)
    needs_macrag_trace = any(
        get_treatment(treatment).retrieval_source == "macrag"
        for treatment in config.treatments
    )

    if needs_dense_trace:
        if dense_retriever is not None:
            for question in questions:
                traces[question.id]["dense"] = dense_retriever.retrieve(
                    question.question,
                    config.top_k,
                )
        else:
            active_dense_retriever = make_dense_retriever(
                config,
                chunks,
                include_reranker=False,
            )
            try:
                for question in questions:
                    traces[question.id]["dense"] = active_dense_retriever.retrieve(
                        question.question,
                        config.top_k,
                    )
            finally:
                del active_dense_retriever
                clear_cuda_cache()

            if config.reranker_model:
                chunk_lookup = chunks_by_id(chunks)
                reranker = make_reranker(config)
                try:
                    for question in questions:
                        traces[question.id]["dense"] = rerank_dense_trace(
                            question.question,
                            traces[question.id]["dense"],
                            chunk_lookup,
                            reranker,
                        )
                finally:
                    del reranker
                    clear_cuda_cache()

    if needs_sparse_trace:
        active_sparse_retriever = sparse_retriever or make_sparse_retriever(
            config,
            chunks,
        )
        try:
            for question in questions:
                traces[question.id]["sparse"] = active_sparse_retriever.retrieve(
                    question.question,
                    config.top_k,
                )
        finally:
            if sparse_retriever is None:
                del active_sparse_retriever

    if needs_macrag_trace:
        from hamlet_qa.features.macrag.retrieval import build_macrag_traces

        build_macrag_traces(
            config,
            chunks,
            questions,
            traces,
            feature_handles=feature_handles,
        )

    return traces


def prepare_crag_handles(
    config: RunConfig,
    chunks: list[dict[str, Any]],
    handles: dict[str, Any],
) -> None:
    """Construct the CRAG strip evaluator and doc-wide BM25 re-retriever.

    The BM25 retriever is CPU-only. The strip evaluator reuses the configured
    reranker model; on the single-GPU layout it runs on CPU so it can stay
    resident next to the vLLM reader (small workload, documented slowness),
    while multi-GPU layouts place it on the reranker device.
    """
    if "crag_reretriever" not in handles:
        handles["crag_reretriever"] = make_sparse_retriever(config, chunks)
    if "crag_evaluator" not in handles:
        if not config.reranker_model:
            raise ValueError(
                "The crag treatment uses the reranker model as its retrieval "
                "evaluator; configure --reranker-model."
            )
        device = config.crag_evaluator_device or (
            "cpu" if config.gpu_layout == "single" else config.reranker_device
        )
        handles["crag_evaluator"] = CrossEncoderReranker(
            config.reranker_model,
            device=device,
            batch_size=config.reranker_batch_size,
        )


def feature_params_from_config(config: RunConfig) -> dict[str, Any]:
    """Per-treatment scalars passed through ContextAssemblyRequest."""
    cache_dir = Path(config.context_assembly_cache_dir)
    return {
        "crag_ndocs": config.crag_ndocs,
        "crag_upper_threshold": config.crag_upper_threshold,
        "crag_lower_threshold": config.crag_lower_threshold,
        "crag_decompose_mode": config.crag_decompose_mode,
        "crag_external_top_k": config.crag_external_top_k,
        "crag_rewrite_cache_path": cache_dir / "crag_rewrite_cache.json",
        "macrag_top_k2": config.macrag_top_k2,
        "macrag_chunk_ext": config.macrag_chunk_ext,
        "macrag_merge_version": config.macrag_merge_version,
        "recomp_input_docs": config.recomp_input_docs,
        "recomp_top_sentences": config.recomp_top_sentences,
        "recomp_abstractive_mode": config.recomp_abstractive_mode,
        "recomp_cache_path": cache_dir / "recomp_cache.json",
    }


def recomp_input_chunks_for_question(
    config: RunConfig,
    question_trace: list[dict[str, Any]],
    chunk_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    chunk_ids = [
        str(row["chunk_id"])
        for row in question_trace[: config.recomp_input_docs]
        if str(row["chunk_id"]) in chunk_lookup
    ]
    return [dict(chunk_lookup[chunk_id]) for chunk_id in chunk_ids]


def precompute_recomp_summaries(
    config: RunConfig,
    questions: list[Question],
    retrieval_traces: dict[str, dict[str, list[dict[str, Any]]]],
    chunk_lookup: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Run the RECOMP compressors before the reader claims the GPU.

    Compression depends only on (question, dense trace), so all summaries are
    computed and cached here, and the compressor models are unloaded again.
    The prompted-Qwen abstractive mode is excluded: it runs at assembly time
    against the resident reader.
    """
    from hamlet_qa.core.llm_cache import JsonKVCache, stable_hash
    from hamlet_qa.features.recomp.compressor import (
        AbstractiveCompressor,
        ExtractiveCompressor,
        compress_abstractive_t5,
        compress_extractive,
    )

    needs_extractive = "recomp_extractive" in config.treatments
    needs_t5_abstractive = (
        "recomp_abstractive" in config.treatments
        and config.recomp_abstractive_mode == "t5"
    )
    if not (needs_extractive or needs_t5_abstractive):
        return {}

    cache = JsonKVCache(
        Path(config.context_assembly_cache_dir) / "recomp_cache.json",
        section="recomp_precomputed",
    )
    summaries: dict[str, dict[str, Any]] = {}

    def inputs_for(question: Question) -> list[dict[str, Any]]:
        question_trace = retrieval_traces.get(question.id, {}).get("dense") or []
        return recomp_input_chunks_for_question(config, question_trace, chunk_lookup)

    def run_stage(
        treatment: str,
        model_name: str,
        make_model: Any,
        compress: Any,
    ) -> None:
        pending: dict[str, tuple[Question, list[dict[str, Any]], str]] = {}
        for question in questions:
            input_chunks = inputs_for(question)
            cache_key = stable_hash(
                {
                    "treatment": treatment,
                    "question": question.question,
                    "input_texts": [chunk["text"] for chunk in input_chunks],
                    "model": model_name,
                    "top_sentences": config.recomp_top_sentences,
                }
            )
            cached = cache.get(cache_key)
            if cached is not None:
                summaries[f"{treatment}:{question.id}"] = dict(cached, cache_hit=True)
            else:
                pending[question.id] = (question, input_chunks, cache_key)
        if not pending:
            return
        model = make_model()
        try:
            for question, input_chunks, cache_key in pending.values():
                record = compress(model, question, input_chunks)
                cache.set(cache_key, record)
                summaries[f"{treatment}:{question.id}"] = dict(record, cache_hit=False)
        finally:
            del model
            clear_cuda_cache()
        cache.save()

    if needs_extractive:
        run_stage(
            "recomp_extractive",
            config.recomp_extractive_model,
            lambda: ExtractiveCompressor(
                config.recomp_extractive_model,
                device=config.embedding_device,
            ),
            lambda model, question, input_chunks: compress_extractive(
                question.question,
                input_chunks,
                model,
                config.recomp_top_sentences,
            ),
        )
    if needs_t5_abstractive:
        run_stage(
            "recomp_abstractive",
            config.recomp_abstractive_model,
            lambda: AbstractiveCompressor(
                config.recomp_abstractive_model,
                device=config.embedding_device,
            ),
            lambda model, question, input_chunks: compress_abstractive_t5(
                question.question,
                input_chunks,
                model,
            ),
        )
    return summaries


def prepare_run_dir(config: RunConfig, questions: list[Question]) -> Path:
    run_dir = config.run_dir
    if run_dir.exists():
        if not config.overwrite:
            raise FileExistsError(
                f"Run directory already exists: {run_dir}. Use --overwrite to replace it."
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    dump_json(run_dir / "run_config.json", config.to_dict())
    shutil.copy2(config.chunks_path, run_dir / "hamlet_chunks.jsonl")
    shutil.copy2(config.questions_path, run_dir / "hamlet_questions_input.json")
    domain_kg_path = Path(config.domain_kg_path)
    if domain_kg_path.exists():
        shutil.copy2(domain_kg_path, run_dir / "hamlet_domain_kg.yaml")
    dump_json(
        run_dir / "hamlet_questions_resolved.json",
        [question.to_dict(include_matches=True) for question in questions],
    )
    return run_dir


def run_experiment(
    config: RunConfig,
    reader: ReaderLike | None = None,
    dense_retriever: RetrieverLike | None = None,
    sparse_retriever: RetrieverLike | None = None,
    feature_handles: dict[str, Any] | None = None,
) -> Path:
    unknown_treatments = sorted(set(config.treatments) - known_treatment_names())
    if unknown_treatments:
        raise ValueError(f"Unknown treatments: {unknown_treatments}")

    chunks = load_jsonl(config.chunks_path)
    lookup = chunks_by_id(chunks)
    doc_order_ids = document_order_chunk_ids(chunks)
    questions = load_questions(config.questions_path)
    validate_questions(questions, chunks)
    domain_kg = None
    if treatments_using_domain_kg(config.treatments):
        domain_kg = DomainKnowledgeGraph.from_file(config.domain_kg_path)

    run_dir = prepare_run_dir(config, questions)
    results_path = run_dir / "results.jsonl"
    retrieval_traces = build_retrieval_traces(
        config,
        chunks,
        questions,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        feature_handles=feature_handles,
    )

    active_handles = dict(feature_handles or {})
    if "recomp_summaries" not in active_handles:
        active_handles["recomp_summaries"] = precompute_recomp_summaries(
            config,
            questions,
            retrieval_traces,
            lookup,
        )
    if "crag" in config.treatments:
        prepare_crag_handles(config, chunks, active_handles)

    needs_reader_for_assembly = treatments_using_llm_assembly(config.treatments) or (
        "recomp_abstractive" in config.treatments
        and config.recomp_abstractive_mode == "prompted_qwen"
    )
    active_reader = reader or make_reader(
        config,
        force_generation_model=needs_reader_for_assembly,
    )
    feature_params = feature_params_from_config(config)

    for question in questions:
        for context_budget in config.context_budgets:
            for treatment in config.treatments:
                retrieval_trace = None
                spec = get_treatment(treatment)
                if spec.retrieval_source == "macrag":
                    retrieval_trace = retrieval_traces[question.id].get("macrag")
                elif treatment in DENSE_TREATMENTS:
                    retrieval_trace = retrieval_traces[question.id].get("dense")
                elif treatment in SPARSE_TREATMENTS:
                    retrieval_trace = retrieval_traces[question.id].get("sparse")
                prepared = prepare_treatment(
                    question,
                    treatment,
                    context_budget,
                    lookup,
                    doc_order_ids,
                    retrieval_trace=retrieval_trace,
                    random_seed=config.random_seed,
                    domain_kg=domain_kg,
                    selector_model=active_reader,
                    setr_cache_path=(
                        Path(config.context_assembly_cache_dir) / "setr_selector_cache.json"
                        if treatment == "setr"
                        else None
                    ),
                    setr_max_passages=config.setr_max_passages,
                    setr_selector_max_tokens=config.setr_selector_max_tokens,
                    feature_params=feature_params,
                    feature_handles=active_handles,
                )
                prompt_bundle = HamletQAPromptBuilder().build(
                    question.question,
                    prepared["selected_chunks"],
                    treatment,
                    active_reader,
                )
                model_output = None
                if not config.prepare_only:
                    model_output = active_reader.generate(
                        prompt_bundle.system_prompt,
                        prompt_bundle.user_prompt,
                    )
                row = build_result_row(
                    question,
                    treatment,
                    context_budget,
                    prepared,
                    active_reader,
                    config,
                    model_output,
                    prompt_bundle=prompt_bundle,
                )
                append_jsonl(results_path, row)

    return results_path
