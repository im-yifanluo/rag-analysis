"""Question-set loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hamlet_qa.config import QUESTION_CATEGORIES
from hamlet_qa.io_utils import load_json


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    question: str
    expected_answer: str
    gold_chunk_ids: list[str]
    notes: str

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Question":
        required = {
            "id",
            "category",
            "question",
            "expected_answer",
            "gold_chunk_ids",
            "notes",
        }
        missing = required - set(row)
        if missing:
            raise ValueError(f"Question is missing fields: {sorted(missing)}")
        if not isinstance(row["gold_chunk_ids"], list):
            raise ValueError(f"{row['id']}: gold_chunk_ids must be a list")
        return cls(
            id=str(row["id"]),
            category=str(row["category"]),
            question=str(row["question"]),
            expected_answer=str(row["expected_answer"]),
            gold_chunk_ids=[str(item) for item in row["gold_chunk_ids"]],
            notes=str(row["notes"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "gold_chunk_ids": list(self.gold_chunk_ids),
            "notes": self.notes,
        }


def load_questions(path: str) -> list[Question]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("Question file must contain a JSON list.")
    return [Question.from_dict(row) for row in data]


def validate_questions(
    questions: list[Question],
    valid_chunk_ids: set[str],
    categories: list[str] | None = None,
) -> None:
    expected_categories = categories or QUESTION_CATEGORIES
    seen_ids: set[str] = set()
    seen_categories: set[str] = set()

    for question in questions:
        if question.id in seen_ids:
            raise ValueError(f"Duplicate question id: {question.id}")
        seen_ids.add(question.id)

        if question.category not in expected_categories:
            raise ValueError(
                f"{question.id}: unknown category {question.category!r}"
            )
        seen_categories.add(question.category)

        unknown = sorted(set(question.gold_chunk_ids) - valid_chunk_ids)
        if unknown:
            raise ValueError(f"{question.id}: unknown gold chunk ids: {unknown}")

        if question.category == "unanswerable" and question.gold_chunk_ids:
            raise ValueError(f"{question.id}: unanswerable questions must use no gold chunks")

    missing_categories = sorted(set(expected_categories) - seen_categories)
    if missing_categories:
        raise ValueError(f"Question file is missing categories: {missing_categories}")
