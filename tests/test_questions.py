from __future__ import annotations

import unittest
from pathlib import Path

from hamlet_qa.config import QUESTION_CATEGORIES
from hamlet_qa.io_utils import load_jsonl
from hamlet_qa.questions import load_questions, validate_questions


REPO_ROOT = Path(__file__).resolve().parents[1]


class QuestionSchemaTests(unittest.TestCase):
    def test_default_questions_cover_categories_and_reference_valid_chunks(self):
        chunks = load_jsonl(REPO_ROOT / "data" / "hamlet_chunks.jsonl")
        questions = load_questions(str(REPO_ROOT / "data" / "hamlet_questions.json"))

        validate_questions(questions, {chunk["chunk_id"] for chunk in chunks})
        self.assertEqual(
            {question.category for question in questions},
            set(QUESTION_CATEGORIES),
        )
        self.assertEqual(len(questions), len(QUESTION_CATEGORIES))

    def test_unanswerable_question_has_no_gold_chunks(self):
        questions = load_questions(str(REPO_ROOT / "data" / "hamlet_questions.json"))
        unanswerable = [
            question
            for question in questions
            if question.category == "unanswerable"
        ]
        self.assertEqual(len(unanswerable), 1)
        self.assertEqual(unanswerable[0].gold_chunk_ids, [])


if __name__ == "__main__":
    unittest.main()
