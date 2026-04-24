"""Experiment orchestration, treatment selection, and result logging."""

from __future__ import annotations

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
from hamlet_qa.retrieval import DenseRetriever, SnowflakeEmbedder


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


def relevance_rank_map(dense_trace: list[dict[str, Any]] | None) -> dict[str, int]:
    if dense_trace is None:
        return {}
    return {
        str(row["chunk_id"]): int(row.get("rank", index + 1))
        for index, row in enumerate(dense_trace)
    }


def sort_by_relevance(
    chunk_ids: list[str],
    chunk_lookup: dict[str, dict[str, Any]],
    dense_trace: list[dict[str, Any]] | None,
) -> list[str]:
    ranks = relevance_rank_map(dense_trace)
    return sorted(
        dedupe_preserve_order(chunk_ids),
        key=lambda chunk_id: (
            ranks.get(chunk_id, 1_000_000),
            int(chunk_lookup[chunk_id]["global_index"]),
        ),
    )


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


def chunk_id_by_global_index(chunks: list[dict[str, Any]]) -> dict[int, str]:
    return {int(chunk["global_index"]): str(chunk["chunk_id"]) for chunk in chunks}


def neighbor_chunk_ids(
    chunk_id: str,
    chunk_lookup: dict[str, dict[str, Any]],
    global_index_lookup: dict[int, str],
    neighbor_window: int,
) -> list[str]:
    center = int(chunk_lookup[chunk_id]["global_index"])
    ids: list[str] = []
    for offset in range(-neighbor_window, neighbor_window + 1):
        if offset == 0:
            continue
        neighbor = global_index_lookup.get(center + offset)
        if neighbor is not None:
            ids.append(neighbor)
    return ids


def local_block_with_neighbors(
    chunk_id: str,
    chunk_lookup: dict[str, dict[str, Any]],
    global_index_lookup: dict[int, str],
    neighbor_window: int,
) -> list[str]:
    center = int(chunk_lookup[chunk_id]["global_index"])
    ids: list[str] = []
    for offset in range(-neighbor_window, neighbor_window + 1):
        neighbor = global_index_lookup.get(center + offset)
        if neighbor is not None:
            ids.append(neighbor)
    return ids


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
    dense_trace: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if dense_trace is None:
        return []
    score_by_id = {
        str(row["chunk_id"]): {
            "chunk_id": str(row["chunk_id"]),
            "rank": int(row["rank"]),
            "score": row.get("score"),
        }
        for row in dense_trace
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
    global_index_lookup: dict[int, str] | None = None,
    dense_trace: list[dict[str, Any]] | None = None,
    neighbor_window: int = 1,
) -> dict[str, Any]:
    del doc_order_ids
    global_index_lookup = global_index_lookup or {
        int(chunk["global_index"]): chunk_id
        for chunk_id, chunk in chunk_lookup.items()
    }
    retrieval_trace = [dict(row) for row in dense_trace] if dense_trace else []
    original_hit_chunk_ids: list[str] = []
    neighbor_chunk_ids_added: list[str] = []

    if treatment == "closed_book":
        candidate_ids: list[str] = []
        prompt_order = "none"
    elif treatment == "gold_evidence":
        original_hit_chunk_ids = list(question.derived_gold_chunk_ids)
        candidate_ids = sort_by_relevance(
            original_hit_chunk_ids,
            chunk_lookup,
            dense_trace,
        )
        prompt_order = "gold_chunks_by_relevance"
    elif treatment == "gold_evidence_neighbors":
        original_hit_chunk_ids = sort_by_relevance(
            list(question.derived_gold_chunk_ids),
            chunk_lookup,
            dense_trace,
        )
        selected_gold = select_chunk_ids_for_budget(
            original_hit_chunk_ids,
            chunk_lookup,
            context_budget,
        )
        neighbor_candidates: list[str] = []
        for chunk_id in original_hit_chunk_ids:
            neighbor_candidates.extend(
                neighbor_chunk_ids(
                    chunk_id,
                    chunk_lookup,
                    global_index_lookup,
                    neighbor_window,
                )
            )
        neighbor_candidates = [
            chunk_id
            for chunk_id in sort_by_relevance(neighbor_candidates, chunk_lookup, dense_trace)
            if chunk_id not in selected_gold
        ]
        remaining_budget = context_budget - context_token_count(
            selected_chunks(selected_gold, chunk_lookup)
        )
        selected_neighbors = select_chunk_ids_for_budget(
            neighbor_candidates,
            chunk_lookup,
            remaining_budget,
        )
        neighbor_chunk_ids_added = selected_neighbors
        candidate_ids = sort_by_relevance(
            selected_gold + selected_neighbors,
            chunk_lookup,
            dense_trace,
        )
        prompt_order = "gold_then_neighbors_by_relevance"
    elif treatment == "dense_relevance":
        if dense_trace is None:
            raise ValueError("dense_relevance requires a dense retrieval trace")
        original_hit_chunk_ids = [str(row["chunk_id"]) for row in dense_trace]
        candidate_ids = original_hit_chunk_ids
        prompt_order = "retrieval_score"
    elif treatment == "dense_relevance_neighbors":
        if dense_trace is None:
            raise ValueError("dense_relevance_neighbors requires a dense retrieval trace")
        original_hit_chunk_ids = [str(row["chunk_id"]) for row in dense_trace]
        blocks: list[str] = []
        neighbors_seen: list[str] = []
        for hit_id in original_hit_chunk_ids:
            block = local_block_with_neighbors(
                hit_id,
                chunk_lookup,
                global_index_lookup,
                neighbor_window,
            )
            for block_id in block:
                blocks.append(block_id)
                if block_id != hit_id:
                    neighbors_seen.append(block_id)
        candidate_ids = dedupe_preserve_order(blocks)
        prompt_order = "retrieval_rank_local_neighbor_blocks"
        neighbor_chunk_ids_added = [
            chunk_id
            for chunk_id in dedupe_preserve_order(neighbors_seen)
            if chunk_id not in original_hit_chunk_ids
        ]
    else:
        raise ValueError(f"Unknown treatment: {treatment}")

    selected_ids = select_chunk_ids_for_budget(candidate_ids, chunk_lookup, context_budget)
    final_chunks = selected_chunks(selected_ids, chunk_lookup)
    quote_presence = required_quotes_present_in_context(question, selected_ids)
    quote_recall = evidence_quote_recall(quote_presence)
    return {
        "selected_chunk_ids": selected_ids,
        "final_selected_chunk_ids": selected_ids,
        "selected_chunks": final_chunks,
        "context_tokens": context_token_count(final_chunks),
        "retrieval_trace": retrieval_trace,
        "retrieval_scores": retrieval_scores_for(selected_ids, dense_trace),
        "prompt_order": prompt_order,
        "original_hit_chunk_ids": original_hit_chunk_ids,
        "neighbor_chunk_ids_added": [
            chunk_id for chunk_id in neighbor_chunk_ids_added if chunk_id in selected_ids
        ],
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
        "neighbor_chunk_ids_added": prepared["neighbor_chunk_ids_added"],
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
        "retrieval_scores": prepared["retrieval_scores"],
        "retrieval_trace": prepared["retrieval_trace"],
        "prompt_order": prepared["prompt_order"],
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "full_prompt": full_prompt,
        "model_output": model_output,
        "failure_label": None,
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
) -> Path:
    unknown_treatments = sorted(set(config.treatments) - set(DEFAULT_TREATMENTS))
    if unknown_treatments:
        raise ValueError(f"Unknown treatments: {unknown_treatments}")

    chunks = load_jsonl(config.chunks_path)
    lookup = chunks_by_id(chunks)
    doc_order_ids = document_order_chunk_ids(chunks)
    global_index_lookup = chunk_id_by_global_index(chunks)
    questions = load_questions(config.questions_path)
    validate_questions(questions, chunks)

    run_dir = prepare_run_dir(config, questions)
    results_path = run_dir / "results.jsonl"
    active_reader = reader or make_reader(config)
    active_retriever = dense_retriever

    needs_dense_trace = any(treatment != "closed_book" for treatment in config.treatments)
    for question in questions:
        dense_trace: list[dict[str, Any]] | None = None
        if needs_dense_trace:
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
                    global_index_lookup=global_index_lookup,
                    dense_trace=dense_trace,
                    neighbor_window=config.neighbor_window,
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
