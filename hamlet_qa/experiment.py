"""Experiment orchestration, treatment selection, and result logging."""

from __future__ import annotations

import gc
import hashlib
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from hamlet_qa.config import DEFAULT_TREATMENTS, RunConfig
from hamlet_qa.generation import VLLMReader
from hamlet_qa.io_utils import append_jsonl, dump_json, load_jsonl
from hamlet_qa.prompts import (
    CLOSED_BOOK_SYSTEM_PROMPT,
    GROUNDED_SYSTEM_PROMPT,
    TokenizerPromptFormatter,
    build_user_prompt,
)
from hamlet_qa.questions import Question, load_questions, validate_questions
from hamlet_qa.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    SentenceTransformerEmbedder,
)


DENSE_TREATMENTS = {
    "dense_reranked",
    "dense_document_order",
    "dense_random_order",
}
SPARSE_TREATMENTS = {"sparse_bm25"}


class ReaderLike(Protocol):
    model_max_context: int | None

    def format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        ...

    def count_tokens(self, text: str) -> int:
        ...

    def generate(self, system_prompt: str, user_prompt: str) -> str:
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


def chunks_by_id(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {chunk["chunk_id"]: dict(chunk) for chunk in chunks}


def document_order_chunk_ids(chunks: list[dict[str, Any]]) -> list[str]:
    return [
        chunk["chunk_id"]
        for chunk in sorted(chunks, key=lambda item: int(item["global_index"]))
    ]


def dedupe_preserve_order(chunk_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for chunk_id in chunk_ids:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        deduped.append(chunk_id)
    return deduped


def sort_by_document_order(
    chunk_ids: list[str],
    chunk_lookup: dict[str, dict[str, Any]],
) -> list[str]:
    return sorted(
        dedupe_preserve_order(chunk_ids),
        key=lambda chunk_id: int(chunk_lookup[chunk_id]["global_index"]),
    )


def stable_random_order(chunk_ids: list[str], random_seed: int, salt: str) -> list[str]:
    ordered = list(chunk_ids)
    seed_material = f"{random_seed}:{salt}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    random.Random(seed).shuffle(ordered)
    return ordered


def select_chunk_ids_for_budget(
    candidate_chunk_ids: list[str],
    chunk_lookup: dict[str, dict[str, Any]],
    context_budget: int,
) -> list[str]:
    selected: list[str] = []
    total_tokens = 0
    for chunk_id in dedupe_preserve_order(candidate_chunk_ids):
        chunk = chunk_lookup[chunk_id]
        token_count = int(chunk["token_count"])
        if total_tokens + token_count > context_budget:
            continue
        selected.append(chunk_id)
        total_tokens += token_count
    return selected


def selected_chunks(
    selected_chunk_ids: list[str],
    chunk_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [dict(chunk_lookup[chunk_id]) for chunk_id in selected_chunk_ids]


def context_token_count(chunks: list[dict[str, Any]]) -> int:
    return sum(int(chunk["token_count"]) for chunk in chunks)


def required_quotes_present_in_context(
    question: Question,
    selected_chunk_ids: list[str],
) -> list[dict[str, Any]]:
    selected = set(selected_chunk_ids)
    present: list[dict[str, Any]] = []
    for index, evidence_quote in enumerate(question.required_evidence_quotes):
        matched = list(evidence_quote.matched_chunk_ids)
        present.append(
            {
                "quote_index": index,
                "quote": evidence_quote.quote,
                "role": evidence_quote.role,
                "matched_chunk_ids": matched,
                "present": bool(selected & set(matched)),
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
) -> dict[str, Any]:
    del doc_order_ids
    trace = [dict(row) for row in retrieval_trace] if retrieval_trace else []
    original_hit_chunk_ids: list[str] = []
    retrieval_method = "none"

    if treatment == "closed_book":
        selected_ids: list[str] = []
        prompt_order = "none"
    elif treatment == "gold_evidence":
        original_hit_chunk_ids = sort_by_document_order(
            list(question.derived_gold_chunk_ids),
            chunk_lookup,
        )
        selected_ids = select_chunk_ids_for_budget(
            original_hit_chunk_ids,
            chunk_lookup,
            context_budget,
        )
        prompt_order = "gold_chunks_document_order"
        retrieval_method = "gold"
    elif treatment in DENSE_TREATMENTS:
        if retrieval_trace is None:
            raise ValueError(f"{treatment} requires a dense retrieval trace")
        original_hit_chunk_ids = [str(row["chunk_id"]) for row in trace]
        reranked_selection = select_chunk_ids_for_budget(
            original_hit_chunk_ids,
            chunk_lookup,
            context_budget,
        )
        retrieval_method = (
            str(trace[0].get("retrieval_method", "dense_faiss_reranked"))
            if trace
            else "dense_faiss_reranked"
        )
        if treatment == "dense_reranked":
            selected_ids = reranked_selection
            prompt_order = "dense_reranker_rank"
        elif treatment == "dense_document_order":
            selected_ids = sort_by_document_order(reranked_selection, chunk_lookup)
            prompt_order = "dense_hits_document_order"
        else:
            selected_ids = stable_random_order(
                reranked_selection,
                random_seed,
                f"{question.id}:{context_budget}:{treatment}",
            )
            prompt_order = "dense_hits_random_order"
    elif treatment in SPARSE_TREATMENTS:
        if retrieval_trace is None:
            raise ValueError(f"{treatment} requires a sparse retrieval trace")
        original_hit_chunk_ids = [str(row["chunk_id"]) for row in trace]
        selected_ids = select_chunk_ids_for_budget(
            original_hit_chunk_ids,
            chunk_lookup,
            context_budget,
        )
        prompt_order = "bm25_score"
        retrieval_method = (
            str(trace[0].get("retrieval_method", "bm25")) if trace else "bm25"
        )
    else:
        raise ValueError(f"Unknown treatment: {treatment}")

    final_chunks = selected_chunks(selected_ids, chunk_lookup)
    quote_presence = required_quotes_present_in_context(question, selected_ids)
    quote_recall = evidence_quote_recall(quote_presence)
    return {
        "selected_chunk_ids": selected_ids,
        "final_selected_chunk_ids": selected_ids,
        "selected_chunks": final_chunks,
        "context_tokens": context_token_count(final_chunks),
        "retrieval_trace": trace,
        "retrieval_scores": retrieval_scores_for(selected_ids, trace),
        "retrieval_method": retrieval_method,
        "prompt_order": prompt_order,
        "original_hit_chunk_ids": original_hit_chunk_ids,
        "evidence_chunk_recall": evidence_chunk_recall(
            selected_ids,
            question.derived_gold_chunk_ids,
        ),
        "evidence_quote_recall": quote_recall,
        "required_quotes_present_in_context": quote_presence,
        "coverage_ratio": (
            context_token_count(final_chunks) / context_budget
            if context_budget > 0
            else None
        ),
    }


def system_prompt_for_treatment(treatment: str) -> str:
    if treatment == "closed_book":
        return CLOSED_BOOK_SYSTEM_PROMPT
    return GROUNDED_SYSTEM_PROMPT


def count_prompt_tokens(reader: Any, full_prompt: str) -> int:
    count_tokens = getattr(reader, "count_tokens", None)
    if callable(count_tokens):
        return int(count_tokens(full_prompt))
    return len(full_prompt.split())


def build_result_row(
    question: Question,
    treatment: str,
    context_budget: int,
    prepared: dict[str, Any],
    reader: Any,
    config: RunConfig,
    model_output: str | None,
) -> dict[str, Any]:
    system_prompt = system_prompt_for_treatment(treatment)
    user_prompt = build_user_prompt(
        question.question,
        prepared["selected_chunks"],
        closed_book=treatment == "closed_book",
    )
    full_prompt = reader.format_prompt(system_prompt, user_prompt)
    prompt_tokens = count_prompt_tokens(reader, full_prompt)
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
        "prompt_tokens": prompt_tokens,
        "total_input_tokens": prompt_tokens,
        "max_new_tokens": config.max_new_tokens,
        "model_name": config.reader_model,
        "model_max_context": getattr(reader, "model_max_context", None),
        "coverage_ratio": prepared["coverage_ratio"],
        "retrieval_method": prepared["retrieval_method"],
        "retrieval_scores": prepared["retrieval_scores"],
        "retrieval_trace": prepared["retrieval_trace"],
        "prompt_order": prepared["prompt_order"],
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "full_prompt": full_prompt,
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


def make_reader(config: RunConfig) -> Any:
    if config.prepare_only:
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
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    traces: dict[str, dict[str, list[dict[str, Any]]]] = {
        question.id: {} for question in questions
    }
    needs_dense_trace = any(treatment in DENSE_TREATMENTS for treatment in config.treatments)
    needs_sparse_trace = any(treatment in SPARSE_TREATMENTS for treatment in config.treatments)

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

    return traces


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
) -> Path:
    unknown_treatments = sorted(set(config.treatments) - set(DEFAULT_TREATMENTS))
    if unknown_treatments:
        raise ValueError(f"Unknown treatments: {unknown_treatments}")

    chunks = load_jsonl(config.chunks_path)
    lookup = chunks_by_id(chunks)
    doc_order_ids = document_order_chunk_ids(chunks)
    questions = load_questions(config.questions_path)
    validate_questions(questions, chunks)

    run_dir = prepare_run_dir(config, questions)
    results_path = run_dir / "results.jsonl"
    retrieval_traces = build_retrieval_traces(
        config,
        chunks,
        questions,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
    )
    active_reader = reader or make_reader(config)

    for question in questions:
        for context_budget in config.context_budgets:
            for treatment in config.treatments:
                retrieval_trace = None
                if treatment in DENSE_TREATMENTS:
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
                )
                system_prompt = system_prompt_for_treatment(treatment)
                user_prompt = build_user_prompt(
                    question.question,
                    prepared["selected_chunks"],
                    closed_book=treatment == "closed_book",
                )
                model_output = None
                if not config.prepare_only:
                    model_output = active_reader.generate(system_prompt, user_prompt)
                row = build_result_row(
                    question,
                    treatment,
                    context_budget,
                    prepared,
                    active_reader,
                    config,
                    model_output,
                )
                append_jsonl(results_path, row)

    return results_path
