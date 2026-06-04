"""LLM-backed SetR selector using the original selection_IRI prompt."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from pathlib import Path
from typing import Any, Protocol

from hamlet_qa.core.context import dedupe_existing_chunk_ids


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


class SetRJsonCache:
    """JSON cache for selector prompts, raw outputs, and parsed selections."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.data: dict[str, Any] = {"selector_outputs": {}}
        if self.path and self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                self.data.update(loaded)

    def get(self, cache_key: str) -> dict[str, Any] | None:
        section = self.data.setdefault("selector_outputs", {})
        if not isinstance(section, dict):
            self.data["selector_outputs"] = {}
            return None
        value = section.get(cache_key)
        return dict(value) if isinstance(value, dict) else None

    def set(self, cache_key: str, value: dict[str, Any]) -> None:
        section = self.data.setdefault("selector_outputs", {})
        if not isinstance(section, dict):
            section = {}
            self.data["selector_outputs"] = section
        section[cache_key] = value

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(self.data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def parse_setr_final_selection(output: str, num_candidates: int) -> list[int]:
    match = re.search(
        r"###\s*Final Selection([\w\W]+)",
        output,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise SetRSelectionError(
            "SetR selector output is missing '### Final Selection'."
        )

    selected: list[int] = []
    seen: set[int] = set()
    for raw_number in re.findall(r"\[(\d+)\]", match.group(1)):
        passage_number = int(raw_number)
        if passage_number < 1 or passage_number > num_candidates:
            continue
        if passage_number in seen:
            continue
        seen.add(passage_number)
        selected.append(passage_number)

    if not selected:
        raise SetRSelectionError(
            "SetR selector output did not contain any valid passage numbers."
        )
    return selected


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
    if not selected:
        raise SetRSelectionError(
            "SetR selector chose passages, but none fit the context budget."
        )
    return selected, skipped_over_budget, total_tokens


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

    cached = cache.get(cache_key)
    if cached is not None:
        selector_output = str(cached.get("selector_output", ""))
        selected_positions = [int(item) for item in cached.get("selected_positions", [])]
        cache_hit = True
    else:
        selector_output = call_selector_model(
            selector_model,
            user_prompt,
            max_tokens=selector_max_tokens,
        )
        selected_positions = parse_setr_final_selection(
            selector_output,
            num_candidates=len(candidates),
        )
        cache.set(
            cache_key,
            {
                "selector_model": selector_model_name,
                "selector_system_prompt": SETR_SELECTION_SYS_PROMPT,
                "selector_user_prompt": user_prompt,
                "selector_output": selector_output,
                "selected_positions": selected_positions,
                "selected_chunk_ids": [
                    candidates[position - 1] for position in selected_positions
                ],
            },
        )
        cache.save()
        cache_hit = False

    if not selected_positions:
        selected_positions = parse_setr_final_selection(
            selector_output,
            num_candidates=len(candidates),
        )
    selected_by_model = [candidates[position - 1] for position in selected_positions]
    selected_ids, skipped_over_budget, total_tokens = selected_ids_within_budget(
        selected_by_model,
        chunk_lookup,
        context_budget,
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
            "selector_system_prompt": SETR_SELECTION_SYS_PROMPT,
            "selector_user_prompt": user_prompt,
            "selector_output": selector_output,
            "selected_positions": selected_positions,
            "selected_chunk_ids_before_budget": selected_by_model,
            "selected_chunk_ids": selected_ids,
            "skipped_over_budget_chunk_ids": skipped_over_budget,
        },
    }
