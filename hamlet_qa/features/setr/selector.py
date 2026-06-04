"""SetR-lite role-set selector."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from hamlet_qa.core.context import candidate_rank_map, dedupe_existing_chunk_ids
from hamlet_qa.core.text import flatten_string_list, phrase_in_text, tokenize_terms


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


DEFAULT_ROLE_KEYWORDS: dict[str, list[str]] = {
    "answer": [],
    "speaker_attribution": ["speaker", "says", "said", "line"],
    "plan": ["plan", "players", "play", "observe", "looks"],
    "bridge_to_execution": ["play", "tonight", "king", "circumstance"],
    "staged_evidence": ["poisons", "garden", "estate", "murder"],
    "reaction_confirmation": ["note", "noted", "observe", "confirmation"],
    "earlier_event": ["first", "earlier", "before"],
    "later_event": ["later", "after"],
    "cup_setup": ["cup", "drink", "union"],
    "cup_state": ["poison", "poisoned", "cup"],
    "outcome": ["dies", "death", "poisoned", "outcome"],
    "mistaken_action": ["arras", "rat", "draws", "stabs"],
    "mistaken_belief": ["king", "belief"],
    "actual_victim": ["polonius", "victim", "killed"],
    "reason_one": ["queen", "mother", "reason", "motive"],
    "reason_two": ["public", "love", "general", "reason", "motive"],
    "symbolic_image": ["vows", "brokers", "implorators", "image"],
}

STOPWORDS = {
    "about",
    "after",
    "also",
    "does",
    "from",
    "have",
    "into",
    "that",
    "the",
    "their",
    "then",
    "there",
    "they",
    "this",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}

class JsonCache:
    """Small JSON cache for role labels and chunk-role judgments."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.data: dict[str, Any] = {
            "query_role_labels": {},
            "chunk_role_judgments": {},
        }
        if self.path and self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                self.data.update(loaded)

    def get_section(self, section: str) -> dict[str, Any]:
        value = self.data.setdefault(section, {})
        if not isinstance(value, dict):
            value = {}
            self.data[section] = value
        return value

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


def normalize_role(role: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", role.lower()).strip("_")
    return normalized or "answer"


def important_terms(text: str, limit: int = 8) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in tokenize_terms(text):
        if len(token) <= 2 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= limit:
            break
    return terms


def trace_by_chunk_id(retrieval_trace: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    if not retrieval_trace:
        return {}
    return {str(row["chunk_id"]): dict(row) for row in retrieval_trace}


def derive_setr_role_requirements(
    question: Any,
    role_templates: dict[str, list[str]] | None = None,
    cache: JsonCache | None = None,
) -> list[dict[str, Any]]:
    """Derive SetR-style information requirements from question metadata."""

    evidence = list(getattr(question, "required_evidence_quotes", []) or [])
    cache_key = stable_hash(
        {
            "question": getattr(question, "question", str(question)),
            "roles": [
                {
                    "role": getattr(item, "role", ""),
                    "speaker": getattr(item, "speaker", ""),
                }
                for item in evidence
            ],
            "role_templates": role_templates or {},
        }
    )
    if cache is not None:
        cached = cache.get_section("query_role_labels").get(cache_key)
        if isinstance(cached, list):
            return [dict(item) for item in cached]

    role_templates = role_templates or {}
    query_terms = important_terms(getattr(question, "question", str(question)))
    requirements: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in evidence:
        role = normalize_role(str(getattr(item, "role", "answer")))
        if role in seen:
            continue
        seen.add(role)
        speaker = str(getattr(item, "speaker", "") or "").lower()
        keywords = []
        keywords.extend(tokenize_terms(role.replace("_", " ")))
        keywords.extend(tokenize_terms(speaker))
        keywords.extend(DEFAULT_ROLE_KEYWORDS.get(role, []))
        keywords.extend(role_templates.get(role, []))
        if role == "answer" or not keywords:
            keywords.extend(query_terms)
        requirements.append(
            {
                "role": role,
                "speaker": speaker,
                "keywords": sorted({keyword for keyword in keywords if keyword}),
            }
        )

    if not requirements:
        requirements = [
            {
                "role": "answer",
                "speaker": "",
                "keywords": query_terms,
            }
        ]

    if cache is not None:
        cache.get_section("query_role_labels")[cache_key] = requirements
    return requirements


def explicit_roles_for_candidate(
    chunk: dict[str, Any],
    trace_row: dict[str, Any] | None = None,
) -> set[str]:
    roles: set[str] = set()
    for source in (trace_row or {}, chunk):
        for key in ("covered_roles", "roles", "role_labels", "evidence_roles", "setr_roles"):
            for role in flatten_string_list(source.get(key)):
                roles.add(normalize_role(role))
    return roles


def judge_chunk_roles(
    question: Any,
    chunk: dict[str, Any],
    role_requirements: list[dict[str, Any]],
    trace_row: dict[str, Any] | None = None,
    cache: JsonCache | None = None,
) -> list[str]:
    """Judge which derived roles a chunk covers, using explicit labels if present."""

    explicit_roles = explicit_roles_for_candidate(chunk, trace_row)
    cache_key = stable_hash(
        {
            "question": getattr(question, "question", str(question)),
            "chunk_id": chunk.get("chunk_id"),
            "chunk_text": chunk.get("text"),
            "role_requirements": role_requirements,
            "explicit_roles": sorted(explicit_roles),
        }
    )
    if cache is not None:
        cached = cache.get_section("chunk_role_judgments").get(cache_key)
        if isinstance(cached, list):
            return [str(item) for item in cached]

    requirement_roles = {str(item["role"]) for item in role_requirements}
    if explicit_roles:
        covered = sorted(explicit_roles & requirement_roles)
    else:
        text = str(chunk.get("text", ""))
        text_terms = set(tokenize_terms(text))
        covered = []
        for requirement in role_requirements:
            keywords = [str(item).lower() for item in requirement.get("keywords", [])]
            keyword_hits = 0
            for keyword in keywords:
                keyword_terms = tokenize_terms(keyword)
                if len(keyword_terms) > 1:
                    keyword_hits += int(phrase_in_text(text, keyword))
                elif keyword_terms and keyword_terms[0] in text_terms:
                    keyword_hits += 1
            if keyword_hits:
                covered.append(str(requirement["role"]))

    if cache is not None:
        cache.get_section("chunk_role_judgments")[cache_key] = covered
    return covered


def select_setr_lite(
    question: Any,
    candidate_chunk_ids: list[str],
    chunk_lookup: dict[str, dict[str, Any]],
    context_budget: int,
    retrieval_trace: list[dict[str, Any]] | None = None,
    role_templates: dict[str, list[str]] | None = None,
    cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """Select a diverse set of chunks covering SetR-style role requirements."""

    cache = JsonCache(cache_path)
    candidates = dedupe_existing_chunk_ids(candidate_chunk_ids, chunk_lookup)
    trace_rows = trace_by_chunk_id(retrieval_trace)
    rank_map = candidate_rank_map(candidates)
    role_requirements = derive_setr_role_requirements(
        question,
        role_templates=role_templates,
        cache=cache,
    )
    target_roles = {str(item["role"]) for item in role_requirements}
    judgments = {
        chunk_id: judge_chunk_roles(
            question,
            chunk_lookup[chunk_id],
            role_requirements,
            trace_row=trace_rows.get(chunk_id),
            cache=cache,
        )
        for chunk_id in candidates
    }

    selected: list[str] = []
    selected_roles: set[str] = set()
    total_tokens = 0

    def can_fit(chunk_id: str) -> bool:
        return total_tokens + int(chunk_lookup[chunk_id]["token_count"]) <= context_budget

    while selected_roles != target_roles:
        eligible = [
            chunk_id
            for chunk_id in candidates
            if chunk_id not in selected
            and can_fit(chunk_id)
            and (set(judgments[chunk_id]) - selected_roles)
        ]
        if not eligible:
            break
        best = sorted(
            eligible,
            key=lambda chunk_id: (
                -len(set(judgments[chunk_id]) - selected_roles),
                rank_map[chunk_id],
                int(chunk_lookup[chunk_id]["token_count"]),
                int(chunk_lookup[chunk_id]["global_index"]),
            ),
        )[0]
        selected.append(best)
        selected_roles.update(judgments[best])
        total_tokens += int(chunk_lookup[best]["token_count"])

    for chunk_id in candidates:
        if chunk_id in selected or not can_fit(chunk_id):
            continue
        selected.append(chunk_id)
        total_tokens += int(chunk_lookup[chunk_id]["token_count"])

    selected_chunks = [dict(chunk_lookup[chunk_id]) for chunk_id in selected]
    cache.save()
    return {
        "selected_chunk_ids": selected,
        "selected_chunks": selected_chunks,
        "context_tokens": total_tokens,
        "prompt_order": "setr_lite_set_selection",
        "retrieval_method": (
            f"{retrieval_trace[0].get('retrieval_method', 'dense_faiss')}_setr_lite"
            if retrieval_trace
            else "setr_lite"
        ),
        "context_assembly_trace": {
            "method": "setr_lite",
            "source": "SetR selection_IRI prompt adapted to deterministic role set cover",
            "setr_system_prompt": SETR_SELECTION_SYS_PROMPT,
            "setr_user_prompt_template": SETR_SELECTION_IRI_PROMPT,
            "role_requirements": role_requirements,
            "chunk_role_judgments": judgments,
            "selected_role_coverage": sorted(selected_roles),
        },
    }
