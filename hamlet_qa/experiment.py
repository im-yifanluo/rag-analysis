"""Experiment orchestration and row construction."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from hamlet_qa.config import DEFAULT_TREATMENTS, RunConfig
from hamlet_qa.generation import VLLMReader
from hamlet_qa.io_utils import append_jsonl, dump_json, load_jsonl
from hamlet_qa.prompts import SYSTEM_PROMPT, TokenizerPromptFormatter, build_user_prompt
from hamlet_qa.questions import Question, load_questions, validate_questions
from hamlet_qa.retrieval import DenseRetriever, SnowflakeEmbedder


class ReaderLike(Protocol):
    def format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        ...

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...


class RetrieverLike(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        ...


def chunks_by_id(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {chunk["chunk_id"]: dict(chunk) for chunk in chunks}


def document_order_chunk_ids(chunks: list[dict[str, Any]]) -> list[str]:
    return [
        chunk["chunk_id"]
        for chunk in sorted(chunks, key=lambda item: int(item["global_index"]))
    ]


def select_chunk_ids_for_budget(
    candidate_chunk_ids: list[str],
    chunk_lookup: dict[str, dict[str, Any]],
    context_budget: int,
) -> list[str]:
    selected: list[str] = []
    total_tokens = 0
    for chunk_id in candidate_chunk_ids:
        chunk = chunk_lookup[chunk_id]
        token_count = int(chunk["token_count"])
        if total_tokens + token_count > context_budget:
            continue
        selected.append(chunk_id)
        total_tokens += token_count
    return selected


def compute_evidence_recall(
    selected_chunk_ids: list[str],
    gold_chunk_ids: list[str],
    treatment: str,
) -> float | None:
    if treatment == "closed_book" or not gold_chunk_ids:
        return None
    selected = set(selected_chunk_ids)
    gold = set(gold_chunk_ids)
    return len(selected & gold) / len(gold)


def prepare_treatment(
    question: Question,
    treatment: str,
    context_budget: int,
    chunk_lookup: dict[str, dict[str, Any]],
    doc_order_ids: list[str],
    dense_trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if treatment == "closed_book":
        selected_ids: list[str] = []
        prompt_order = "none"
        retrieval_trace: list[dict[str, Any]] = []
    elif treatment == "gold_evidence":
        gold_set = set(question.gold_chunk_ids)
        candidate_ids = [chunk_id for chunk_id in doc_order_ids if chunk_id in gold_set]
        selected_ids = select_chunk_ids_for_budget(
            candidate_ids,
            chunk_lookup,
            context_budget,
        )
        prompt_order = "document_order_gold"
        retrieval_trace = []
    elif treatment == "dense_relevance":
        if dense_trace is None:
            raise ValueError("dense_relevance requires a dense retrieval trace")
        candidate_ids = [row["chunk_id"] for row in dense_trace]
        selected_ids = select_chunk_ids_for_budget(
            candidate_ids,
            chunk_lookup,
            context_budget,
        )
        prompt_order = "retrieval_score"
        retrieval_trace = [dict(row) for row in dense_trace]
    else:
        raise ValueError(f"Unknown treatment: {treatment}")

    selected_chunks = [dict(chunk_lookup[chunk_id]) for chunk_id in selected_ids]
    context_tokens = sum(int(chunk["token_count"]) for chunk in selected_chunks)
    score_by_id = {
        row["chunk_id"]: row.get("score")
        for row in retrieval_trace
        if row.get("chunk_id") in selected_ids
    }
    retrieval_scores = [
        {"chunk_id": chunk_id, "score": score_by_id[chunk_id]}
        for chunk_id in selected_ids
        if chunk_id in score_by_id
    ]
    return {
        "selected_chunk_ids": selected_ids,
        "selected_chunks": selected_chunks,
        "context_tokens": context_tokens,
        "retrieval_trace": retrieval_trace,
        "retrieval_scores": retrieval_scores,
        "prompt_order": prompt_order,
        "evidence_recall": compute_evidence_recall(
            selected_ids,
            question.gold_chunk_ids,
            treatment,
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
) -> dict[str, Any]:
    user_prompt = build_user_prompt(question.question, prepared["selected_chunks"])
    full_prompt = reader.format_prompt(SYSTEM_PROMPT, user_prompt)
    return {
        "run_name": config.run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "question_id": question.id,
        "category": question.category,
        "question": question.question,
        "expected_answer": question.expected_answer,
        "gold_chunk_ids": list(question.gold_chunk_ids),
        "treatment": treatment,
        "context_budget": context_budget,
        "context_token_count": prepared["context_tokens"],
        "selected_chunk_ids": prepared["selected_chunk_ids"],
        "raw_chunks": prepared["selected_chunks"],
        "evidence_recall": prepared["evidence_recall"],
        "retrieval_scores": prepared["retrieval_scores"],
        "retrieval_trace": prepared["retrieval_trace"],
        "prompt_order": prepared["prompt_order"],
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "full_prompt": full_prompt,
        "model_output": model_output,
        "reader_model": config.reader_model,
        "embedding_model": config.embedding_model,
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
    )


def make_dense_retriever(config: RunConfig, chunks: list[dict[str, Any]]) -> DenseRetriever:
    embedder = SnowflakeEmbedder(
        config.embedding_model,
        device=config.embedding_device,
        batch_size=config.embedding_batch_size,
    )
    return DenseRetriever(embedder, chunks)


def prepare_run_dir(config: RunConfig) -> Path:
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
    shutil.copy2(config.questions_path, run_dir / "hamlet_questions.json")
    return run_dir


def run_experiment(
    config: RunConfig,
    reader: ReaderLike | None = None,
    dense_retriever: RetrieverLike | None = None,
) -> Path:
    unknown_treatments = sorted(set(config.treatments) - set(DEFAULT_TREATMENTS))
    if unknown_treatments:
        raise ValueError(f"Unknown treatments: {unknown_treatments}")

    chunks = load_jsonl(config.chunks_path)
    lookup = chunks_by_id(chunks)
    doc_order_ids = document_order_chunk_ids(chunks)
    questions = load_questions(config.questions_path)
    validate_questions(questions, set(lookup))

    run_dir = prepare_run_dir(config)
    results_path = run_dir / "results.jsonl"
    active_reader = reader or make_reader(config)
    active_retriever = dense_retriever

    for question in questions:
        dense_trace: list[dict[str, Any]] | None = None
        if "dense_relevance" in config.treatments:
            if active_retriever is None:
                active_retriever = make_dense_retriever(config, chunks)
            dense_trace = active_retriever.retrieve(question.question, config.top_k)

        for context_budget in config.context_budgets:
            for treatment in config.treatments:
                prepared = prepare_treatment(
                    question,
                    treatment,
                    context_budget,
                    lookup,
                    doc_order_ids,
                    dense_trace=dense_trace,
                )
                user_prompt = build_user_prompt(
                    question.question,
                    prepared["selected_chunks"],
                )
                model_output = None
                if not config.prepare_only:
                    model_output = active_reader.generate(SYSTEM_PROMPT, user_prompt)
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
