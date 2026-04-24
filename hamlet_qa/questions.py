"""Question-set loading, quote normalization, and evidence derivation."""

from __future__ import annotations

import re
import unicodedata
import warnings
from dataclasses import dataclass, field
from typing import Any

from hamlet_qa.config import REASONING_SKILLS
from hamlet_qa.io_utils import load_json

MANY_MATCH_WARNING_THRESHOLD = 3


@dataclass
class RequiredEvidenceQuote:
    act: int | None
    scene: int | None
    speaker: str
    quote: str
    role: str
    matched_chunk_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "RequiredEvidenceQuote":
        required = {"act", "scene", "speaker", "quote", "role"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"Evidence quote is missing fields: {sorted(missing)}")
        return cls(
            act=None if row["act"] is None else int(row["act"]),
            scene=None if row["scene"] is None else int(row["scene"]),
            speaker=str(row["speaker"]),
            quote=str(row["quote"]),
            role=str(row["role"]),
        )

    def to_dict(self, include_matches: bool = True) -> dict[str, Any]:
        row = {
            "act": self.act,
            "scene": self.scene,
            "speaker": self.speaker,
            "quote": self.quote,
            "role": self.role,
        }
        if include_matches:
            row["matched_chunk_ids"] = list(self.matched_chunk_ids)
        return row


@dataclass
class Question:
    id: str
    question: str
    expected_answer: str
    evidence_scope: str
    reasoning_skill: str
    required_evidence_quotes: list[RequiredEvidenceQuote]
    derived_gold_chunk_ids: list[str]
    notes: str

    @property
    def category(self) -> str:
        return self.reasoning_skill

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Question":
        required = {
            "id",
            "question",
            "expected_answer",
            "evidence_scope",
            "reasoning_skill",
            "required_evidence_quotes",
            "derived_gold_chunk_ids",
            "notes",
        }
        missing = required - set(row)
        if missing:
            raise ValueError(f"Question is missing fields: {sorted(missing)}")
        if not isinstance(row["required_evidence_quotes"], list):
            raise ValueError(f"{row['id']}: required_evidence_quotes must be a list")
        if not isinstance(row["derived_gold_chunk_ids"], list):
            raise ValueError(f"{row['id']}: derived_gold_chunk_ids must be a list")
        return cls(
            id=str(row["id"]),
            question=str(row["question"]),
            expected_answer=str(row["expected_answer"]),
            evidence_scope=str(row["evidence_scope"]),
            reasoning_skill=str(row["reasoning_skill"]),
            required_evidence_quotes=[
                RequiredEvidenceQuote.from_dict(item)
                for item in row["required_evidence_quotes"]
            ],
            derived_gold_chunk_ids=[
                str(item) for item in row["derived_gold_chunk_ids"]
            ],
            notes=str(row["notes"]),
        )

    def to_dict(self, include_matches: bool = True) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "evidence_scope": self.evidence_scope,
            "reasoning_skill": self.reasoning_skill,
            "required_evidence_quotes": [
                quote.to_dict(include_matches=include_matches)
                for quote in self.required_evidence_quotes
            ],
            "derived_gold_chunk_ids": list(self.derived_gold_chunk_ids),
            "notes": self.notes,
        }


def normalize_text(text: str) -> str:
    """Normalize text for quote matching against chunk text."""
    normalized = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def load_questions(path: str) -> list[Question]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("Question file must contain a JSON list.")
    return [Question.from_dict(row) for row in data]


def quote_matches_chunk(quote: RequiredEvidenceQuote, chunk: dict[str, Any]) -> bool:
    if quote.act is not None and int(chunk["act"]) != quote.act:
        return False
    if quote.scene is not None and int(chunk["scene"]) != quote.scene:
        return False
    return normalize_text(quote.quote) in normalize_text(str(chunk["text"]))


def derive_gold_chunk_ids(
    question: Question,
    chunks: list[dict[str, Any]],
    many_match_warning_threshold: int = MANY_MATCH_WARNING_THRESHOLD,
) -> list[str]:
    derived: list[str] = []
    for evidence_quote in question.required_evidence_quotes:
        matches = [
            str(chunk["chunk_id"])
            for chunk in chunks
            if quote_matches_chunk(evidence_quote, chunk)
        ]
        evidence_quote.matched_chunk_ids = matches
        if not matches:
            raise ValueError(
                f"{question.id}: required quote matched zero chunks: "
                f"{evidence_quote.quote!r}"
            )
        if len(matches) > many_match_warning_threshold:
            warnings.warn(
                f"{question.id}: quote matched {len(matches)} chunks: "
                f"{evidence_quote.quote!r}",
                stacklevel=2,
            )
        for chunk_id in matches:
            if chunk_id not in derived:
                derived.append(chunk_id)
    question.derived_gold_chunk_ids = derived
    return derived


def validate_questions(
    questions: list[Question],
    chunks: list[dict[str, Any]],
    reasoning_skills: list[str] | None = None,
) -> list[Question]:
    expected_skills = reasoning_skills or REASONING_SKILLS
    seen_ids: set[str] = set()
    seen_skills: set[str] = set()

    for question in questions:
        if question.id in seen_ids:
            raise ValueError(f"Duplicate question id: {question.id}")
        seen_ids.add(question.id)

        if question.reasoning_skill not in expected_skills:
            raise ValueError(
                f"{question.id}: unknown reasoning_skill {question.reasoning_skill!r}"
            )
        seen_skills.add(question.reasoning_skill)

        if question.reasoning_skill == "unanswerable":
            question.derived_gold_chunk_ids = []
            if question.required_evidence_quotes:
                raise ValueError(
                    f"{question.id}: unanswerable questions must use no required evidence quotes"
                )
            continue

        if not question.required_evidence_quotes:
            raise ValueError(
                f"{question.id}: non-unanswerable questions need required evidence quotes"
            )
        derive_gold_chunk_ids(question, chunks)

    missing_skills = sorted(set(expected_skills) - seen_skills)
    if missing_skills:
        raise ValueError(f"Question file is missing reasoning skills: {missing_skills}")
    return questions
