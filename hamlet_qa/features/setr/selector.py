"""LLM-backed SetR selector using the original selection_IRI prompt."""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any, Protocol

from hamlet_qa.core.context import dedupe_existing_chunk_ids
from hamlet_qa.core.llm_cache import JsonKVCache, stable_hash


SETR_MAX_PASSAGES = 50
SETR_SELECTION_SYS_PROMPT = (
    "You are RankLLM, an intelligent assistant that can rank and select passages "
    "based on their relevancy to the query."
)

SETR_SELECTION_IRI_PROMPT = """I will provide you with {num} passages, each indicated by a numerical identifier []. Select the passages based on their relevance to the search query: {question}.

{context}


Search Query: {question}


Please follow the steps below:
Step 1. Please list up the information requirements to answer the query.
Step 2. for each requirement in Step 1, find the passages that has the information of the requirement.
Step 3. Choose the passages that mostly covers clear and diverse informations to answer the query. Number of passages is unlimited. The format of final output should be '### Final Selection: [] []', e.g., ### Final Selection: [4] [2]."""


class SetRSelectorModel(Protocol):
    model_name: str

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        ...


class SetRSelectionError(ValueError):
    """Raised when the SetR selector model does not produce a usable selection."""


class SetRJsonCache(JsonKVCache):
    """JSON cache for selector prompts, raw outputs, and parsed selections."""

    def __init__(self, path: str | Path | None):
        super().__init__(path, section="selector_outputs")


def format_setr_passage(index: int, chunk: dict[str, Any]) -> str:
    text = re.sub(r"\n+", " ", str(chunk.get("text", "")).strip())
    return f"[{index}] {text}"


def build_setr_user_prompt(
    question_text: str,
    candidate_chunks: list[dict[str, Any]],
) -> str:
    passages = [
        format_setr_passage(index, chunk)
        for index, chunk in enumerate(candidate_chunks, start=1)
    ]
    return SETR_SELECTION_IRI_PROMPT.format(
        question=question_text,
        context="\n\n\n".join(passages),
        num=len(candidate_chunks),
    )


def parse_setr_final_selection(output: str) -> list[int]:
    """Parse the raw selection exactly like the official SetR code.

    Mirrors third_party/SetR/SetR/data_formatting.py: extract everything after
    '### Final Selection' (case-insensitive, optional space after ###), then
    take ALL digit runs — bare digits are accepted, duplicates and
    out-of-range numbers are preserved. Validation against the candidate list
    happens separately in `map_positions_to_chunk_positions`.
    """
    match = re.search(
        r"###\s?Final Selection([\w\W]+)",
        output,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise SetRSelectionError(
            "SetR selector output is missing '### Final Selection'."
        )
    raw_numbers = [int(number) for number in re.findall(r"\d+", match.group(1))]
    if not raw_numbers:
        raise SetRSelectionError(
            "SetR selector output did not contain any passage numbers."
        )
    return raw_numbers


def map_positions_to_chunk_positions(
    raw_positions: list[int],
    num_candidates: int,
) -> dict[str, list[int]]:
    """Map raw selector numbers to usable 1-based candidate positions.

    The official code performs no dedupe or range check; both are necessary
    here to map passage numbers onto chunk IDs, so dropped numbers are
    reported for the assembly trace rather than silently discarded.
    """
    selected: list[int] = []
    seen: set[int] = set()
    dropped_out_of_range: list[int] = []
    dropped_duplicates: list[int] = []
    for position in raw_positions:
        if position < 1 or position > num_candidates:
            dropped_out_of_range.append(position)
            continue
        if position in seen:
            dropped_duplicates.append(position)
            continue
        seen.add(position)
        selected.append(position)
    if not selected:
        raise SetRSelectionError(
            "SetR selector output did not contain any valid passage numbers."
        )
    return {
        "selected_positions": selected,
        "dropped_out_of_range": dropped_out_of_range,
        "dropped_duplicates": dropped_duplicates,
    }


def call_selector_model(
    selector_model: SetRSelectorModel,
    user_prompt: str,
    max_tokens: int,
) -> str:
    signature = inspect.signature(selector_model.generate)
    if "max_tokens" in signature.parameters:
        return selector_model.generate(
            SETR_SELECTION_SYS_PROMPT,
            user_prompt,
            max_tokens=max_tokens,
        )
    return selector_model.generate(SETR_SELECTION_SYS_PROMPT, user_prompt)


def selected_ids_within_budget(
    selected_chunk_ids: list[str],
    chunk_lookup: dict[str, dict[str, Any]],
    context_budget: int,
    allow_empty: bool = False,
) -> tuple[list[str], list[str], int]:
    selected: list[str] = []
    skipped_over_budget: list[str] = []
    total_tokens = 0
    for chunk_id in selected_chunk_ids:
        token_count = int(chunk_lookup[chunk_id]["token_count"])
        if total_tokens + token_count > context_budget:
            skipped_over_budget.append(chunk_id)
            continue
        selected.append(chunk_id)
        total_tokens += token_count
    if not selected and not allow_empty:
        raise SetRSelectionError(
            "SetR selector chose passages, but none fit the context budget."
        )
    return selected, skipped_over_budget, total_tokens


def question_allows_empty_selection(question: Any) -> bool:
    """Allow SetR to abstain for questions with no required evidence quotes."""
    required_quotes = getattr(question, "required_evidence_quotes", None)
    return required_quotes is not None and len(required_quotes) == 0


def select_setr(
    question: Any,
    candidate_chunk_ids: list[str],
    chunk_lookup: dict[str, dict[str, Any]],
    context_budget: int,
    selector_model: SetRSelectorModel,
    retrieval_trace: list[dict[str, Any]] | None = None,
    cache_path: str | Path | None = None,
    selector_max_tokens: int = 4096,
    max_passages: int = SETR_MAX_PASSAGES,
) -> dict[str, Any]:
    """Run SetR selection_IRI with a selector LLM and return selected chunks."""

    candidates = dedupe_existing_chunk_ids(candidate_chunk_ids, chunk_lookup)
    candidates = candidates[:max_passages]
    if not candidates:
        raise SetRSelectionError("SetR requires at least one candidate chunk.")

    question_text = str(getattr(question, "question", str(question)))
    candidate_chunks = [dict(chunk_lookup[chunk_id]) for chunk_id in candidates]
    user_prompt = build_setr_user_prompt(question_text, candidate_chunks)
    selector_model_name = str(getattr(selector_model, "model_name", "reader_model"))
    cache = SetRJsonCache(cache_path)
    cache_key = stable_hash(
        {
            "question": question_text,
            "candidate_chunk_ids": candidates,
            "candidate_texts": [chunk["text"] for chunk in candidate_chunks],
            "selector_model": selector_model_name,
            "selector_prompt": user_prompt,
            "selector_max_tokens": selector_max_tokens,
        }
    )

    allow_empty_selection = question_allows_empty_selection(question)
    parse_error: str | None = None
    cached = cache.get(cache_key)
    if cached is not None and cached.get("raw_selected_numbers") is not None:
        selector_output = str(cached.get("selector_output", ""))
        raw_selected_numbers = [
            int(item) for item in cached.get("raw_selected_numbers", [])
        ]
        cache_hit = True
    else:
        selector_output = call_selector_model(
            selector_model,
            user_prompt,
            max_tokens=selector_max_tokens,
        )
        try:
            raw_selected_numbers = parse_setr_final_selection(selector_output)
        except SetRSelectionError as error:
            if not allow_empty_selection:
                raise
            parse_error = str(error)
            raw_selected_numbers = []
        cache_hit = False

    if raw_selected_numbers:
        mapping = map_positions_to_chunk_positions(
            raw_selected_numbers,
            num_candidates=len(candidates),
        )
    elif allow_empty_selection:
        mapping = {
            "selected_positions": [],
            "dropped_out_of_range": [],
            "dropped_duplicates": [],
        }
    else:
        raise SetRSelectionError(
            "SetR selector output did not contain any valid passage numbers."
        )
    selected_positions = mapping["selected_positions"]
    if not cache_hit:
        cache.set(
            cache_key,
            {
                "selector_model": selector_model_name,
                "selector_system_prompt": SETR_SELECTION_SYS_PROMPT,
                "selector_user_prompt": user_prompt,
                "selector_output": selector_output,
                "raw_selected_numbers": raw_selected_numbers,
                "parse_error": parse_error,
                "empty_selection_allowed": allow_empty_selection,
                "selected_positions": selected_positions,
                "selected_chunk_ids": [
                    candidates[position - 1] for position in selected_positions
                ],
            },
        )
        cache.save()
    selected_by_model = [candidates[position - 1] for position in selected_positions]
    selected_ids, skipped_over_budget, total_tokens = selected_ids_within_budget(
        selected_by_model,
        chunk_lookup,
        context_budget,
        allow_empty=allow_empty_selection,
    )
    return {
        "selected_chunk_ids": selected_ids,
        "selected_chunks": [dict(chunk_lookup[chunk_id]) for chunk_id in selected_ids],
        "context_tokens": total_tokens,
        "prompt_order": "setr_selection_iri",
        "retrieval_method": (
            f"{retrieval_trace[0].get('retrieval_method', 'dense_faiss')}_setr"
            if retrieval_trace
            else "setr"
        ),
        "context_assembly_trace": {
            "method": "setr",
            "source": "third_party/SetR selection_IRI prompt with reader model as selector",
            "selector_model": selector_model_name,
            "selector_max_tokens": selector_max_tokens,
            "max_passages": max_passages,
            "cache_hit": cache_hit,
            "empty_selection_allowed": allow_empty_selection,
            "parse_error": parse_error,
            "selector_system_prompt": SETR_SELECTION_SYS_PROMPT,
            "selector_user_prompt": user_prompt,
            "selector_output": selector_output,
            "raw_selected_numbers": raw_selected_numbers,
            "selected_positions": selected_positions,
            "dropped_out_of_range_numbers": mapping["dropped_out_of_range"],
            "dropped_duplicate_numbers": mapping["dropped_duplicates"],
            "selected_chunk_ids_before_budget": selected_by_model,
            "selected_chunk_ids": selected_ids,
            "skipped_over_budget_chunk_ids": skipped_over_budget,
        },
    }
