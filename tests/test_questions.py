from __future__ import annotations

import unittest
from pathlib import Path

from hamlet_qa.config import REASONING_SKILLS
from hamlet_qa.io_utils import load_jsonl
from hamlet_qa.questions import (
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

        validate_questions(questions, chunks)

        self.assertEqual(
            {question.reasoning_skill for question in questions},
            set(REASONING_SKILLS),
        )
        self.assertEqual(len(questions), len(REASONING_SKILLS))
        answerable = [
            question
            for question in questions
            if question.reasoning_skill != "unanswerable"
        ]
        self.assertTrue(all(question.derived_gold_chunk_ids for question in answerable))

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
