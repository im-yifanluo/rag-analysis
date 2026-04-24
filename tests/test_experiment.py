from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hamlet_qa.config import RunConfig
from hamlet_qa.experiment import (
    build_result_row,
    chunks_by_id,
    document_order_chunk_ids,
    prepare_treatment,
    run_experiment,
)
from hamlet_qa.questions import Question


REPO_ROOT = Path(__file__).resolve().parents[1]


class StubReader:
    def __init__(self):
        self.generate_calls = 0

    def format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        return f"SYSTEM: {system_prompt}\nUSER: {user_prompt}\nASSISTANT:"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        del system_prompt, user_prompt
        self.generate_calls += 1
        raise AssertionError("generate should not be called in prepare-only mode")


def synthetic_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "act01_scene01_chunk001",
            "global_index": 0,
            "act": 1,
            "scene": 1,
            "scene_id": "act01_scene01",
            "scene_title": "First.",
            "chunk_in_scene": 1,
            "start_token": 0,
            "end_token": 3,
            "token_count": 3,
            "text": "alpha beta gamma",
        },
        {
            "chunk_id": "act01_scene01_chunk002",
            "global_index": 1,
            "act": 1,
            "scene": 1,
            "scene_id": "act01_scene01",
            "scene_title": "First.",
            "chunk_in_scene": 2,
            "start_token": 2,
            "end_token": 5,
            "token_count": 3,
            "text": "gamma delta epsilon",
        },
        {
            "chunk_id": "act01_scene02_chunk001",
            "global_index": 2,
            "act": 1,
            "scene": 2,
            "scene_id": "act01_scene02",
            "scene_title": "Second.",
            "chunk_in_scene": 1,
            "start_token": 0,
            "end_token": 2,
            "token_count": 2,
            "text": "zeta eta",
        },
    ]


class SelectionAndPromptTests(unittest.TestCase):
    def setUp(self):
        self.chunks = synthetic_chunks()
        self.lookup = chunks_by_id(self.chunks)
        self.doc_order = document_order_chunk_ids(self.chunks)
        self.question = Question(
            id="q",
            category="local_fact",
            question="What happened?",
            expected_answer="Something.",
            gold_chunk_ids=["act01_scene02_chunk001", "act01_scene01_chunk002"],
            notes="test",
        )

    def test_closed_book_has_empty_context_and_null_recall(self):
        prepared = prepare_treatment(
            self.question,
            "closed_book",
            5,
            self.lookup,
            self.doc_order,
        )

        self.assertEqual(prepared["selected_chunk_ids"], [])
        self.assertEqual(prepared["context_tokens"], 0)
        self.assertIsNone(prepared["evidence_recall"])
        self.assertEqual(prepared["retrieval_scores"], [])
        self.assertEqual(prepared["prompt_order"], "none")

    def test_gold_evidence_uses_document_order_and_budget(self):
        prepared = prepare_treatment(
            self.question,
            "gold_evidence",
            3,
            self.lookup,
            self.doc_order,
        )

        self.assertEqual(prepared["selected_chunk_ids"], ["act01_scene01_chunk002"])
        self.assertEqual(prepared["context_tokens"], 3)
        self.assertEqual(prepared["evidence_recall"], 0.5)
        self.assertEqual(prepared["prompt_order"], "document_order_gold")

    def test_dense_relevance_uses_score_order_and_logs_scores(self):
        trace = [
            {"rank": 1, "chunk_id": "act01_scene02_chunk001", "score": 0.9},
            {"rank": 2, "chunk_id": "act01_scene01_chunk001", "score": 0.8},
            {"rank": 3, "chunk_id": "act01_scene01_chunk002", "score": 0.7},
        ]
        prepared = prepare_treatment(
            self.question,
            "dense_relevance",
            5,
            self.lookup,
            self.doc_order,
            dense_trace=trace,
        )

        self.assertEqual(
            prepared["selected_chunk_ids"],
            ["act01_scene02_chunk001", "act01_scene01_chunk001"],
        )
        self.assertEqual(prepared["context_tokens"], 5)
        self.assertEqual(prepared["evidence_recall"], 0.5)
        self.assertEqual(
            prepared["retrieval_scores"],
            [
                {"chunk_id": "act01_scene02_chunk001", "score": 0.9},
                {"chunk_id": "act01_scene01_chunk001", "score": 0.8},
            ],
        )
        self.assertEqual(prepared["prompt_order"], "retrieval_score")

    def test_result_row_logs_full_prompt_and_chunk_labels(self):
        prepared = prepare_treatment(
            self.question,
            "gold_evidence",
            10,
            self.lookup,
            self.doc_order,
        )
        config = RunConfig(run_name="unit", prepare_only=True)
        row = build_result_row(
            self.question,
            "gold_evidence",
            10,
            prepared,
            StubReader(),
            config,
            model_output=None,
        )

        self.assertIn("act01_scene01_chunk002", row["user_prompt"])
        self.assertIn("SYSTEM:", row["full_prompt"])
        self.assertIsNone(row["model_output"])
        self.assertEqual(row["raw_chunks"][0]["chunk_id"], "act01_scene01_chunk002")


class PrepareOnlyRunTests(unittest.TestCase):
    def test_prepare_only_does_not_call_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader = StubReader()
            config = RunConfig(
                chunks_path=str(REPO_ROOT / "data" / "hamlet_chunks.jsonl"),
                questions_path=str(REPO_ROOT / "data" / "hamlet_questions.json"),
                output_dir=tmp,
                run_name="prepare_only",
                treatments=["closed_book"],
                context_budgets=[500],
                prepare_only=True,
            )

            results_path = run_experiment(config, reader=reader)

            self.assertEqual(reader.generate_calls, 0)
            rows = [
                json.loads(line)
                for line in Path(results_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 11)
            self.assertTrue(all(row["model_output"] is None for row in rows))
            self.assertTrue(all(row["selected_chunk_ids"] == [] for row in rows))


if __name__ == "__main__":
    unittest.main()
