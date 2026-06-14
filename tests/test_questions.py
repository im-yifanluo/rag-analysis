from __future__ import annotations

import unittest
import warnings
from pathlib import Path

from hamlet_qa.core.config import REASONING_SKILLS
from hamlet_qa.core.io import load_jsonl
from hamlet_qa.core.questions import (
    RequiredEvidenceQuote,
    derive_gold_chunk_ids,
    load_questions,
    normalize_text,
    validate_questions,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class QuoteEvidenceTests(unittest.TestCase):
    def test_quote_normalization_collapses_quotes_case_and_whitespace(self):
        self.assertEqual(
            normalize_text("  HAMLET.\nIt’s  “Time”\n"),
            "hamlet. it's \"time\"",
        )

    def test_derived_gold_chunk_ids_generation(self):
        chunks = [
            {"chunk_id": "c1", "act": 1, "scene": 1, "text": "HAMLET.\nHello there."},
            {"chunk_id": "c2", "act": 1, "scene": 2, "text": "HAMLET.\nHello there."},
        ]
        question = load_questions(str(REPO_ROOT / "data" / "hamlet_questions.json"))[0]
        question.required_evidence_quotes = [
            RequiredEvidenceQuote(
                act=1,
                scene=1,
                speaker="HAMLET",
                quote="hamlet. hello there.",
                role="answer",
            )
        ]

        derived = derive_gold_chunk_ids(question, chunks)

        self.assertEqual(derived, ["c1"])
        self.assertEqual(question.required_evidence_quotes[0].matched_chunk_ids, ["c1"])

    def test_default_questions_validate_and_fill_derived_gold(self):
        chunks = load_jsonl(REPO_ROOT / "data" / "hamlet_chunks.jsonl")
        questions = load_questions(str(REPO_ROOT / "data" / "hamlet_questions.json"))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            validate_questions(questions, chunks)

        self.assertEqual(len(questions), 10)
        self.assertTrue(
            {question.reasoning_skill for question in questions}
            <= set(REASONING_SKILLS)
        )
        answerable = [
            question
            for question in questions
            if question.reasoning_skill != "unanswerable"
        ]
        self.assertTrue(all(question.derived_gold_chunk_ids for question in answerable))

    def test_missing_reasoning_skills_warn_instead_of_raising(self):
        chunks = load_jsonl(REPO_ROOT / "data" / "hamlet_chunks.jsonl")
        questions = load_questions(str(REPO_ROOT / "data" / "hamlet_questions.json"))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_questions(questions[:3], chunks)

        messages = [str(item.message) for item in caught]
        self.assertTrue(
            any("does not cover reasoning skills" in message for message in messages)
        )

    def test_budget_pressure_question_needs_more_than_default_budget(self):
        # q_final_scene_deaths is designed so the minimal covering chunk set
        # exceeds the default 1000-token budget; raw-chunk treatments should
        # cap below 1.0 quote recall while compression treatments can pass.
        chunks = load_jsonl(REPO_ROOT / "data" / "hamlet_chunks.jsonl")
        questions = load_questions(str(REPO_ROOT / "data" / "hamlet_questions.json"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            validate_questions(questions, chunks)
        question = next(q for q in questions if q.id == "q_final_scene_deaths")
        lookup = {str(chunk["chunk_id"]): chunk for chunk in chunks}

        cover: set[str] = set()
        for quote in question.required_evidence_quotes:
            if not set(quote.matched_chunk_ids) & cover:
                cover.add(quote.matched_chunk_ids[0])
        cover_tokens = sum(int(lookup[chunk_id]["token_count"]) for chunk_id in cover)

        self.assertGreater(cover_tokens, 1000)

    def test_unanswerable_question_has_no_required_quotes(self):
        questions = load_questions(str(REPO_ROOT / "data" / "hamlet_questions.json"))
        unanswerable = [
            question
            for question in questions
            if question.reasoning_skill == "unanswerable"
        ]
        self.assertEqual(len(unanswerable), 1)
        self.assertEqual(unanswerable[0].required_evidence_quotes, [])


if __name__ == "__main__":
    unittest.main()
