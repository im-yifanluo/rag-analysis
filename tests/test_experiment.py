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
    evidence_quote_recall,
    prepare_treatment,
    required_quotes_present_in_context,
    run_experiment,
)
from hamlet_qa.questions import Question, RequiredEvidenceQuote


REPO_ROOT = Path(__file__).resolve().parents[1]


class StubReader:
    def __init__(self):
        self.generate_calls = 0
        self.model_max_context = 4096

    def format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        return f"SYSTEM: {system_prompt}\nUSER: {user_prompt}\nASSISTANT:"

    def count_tokens(self, text: str) -> int:
        return len(text.split())

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


def synthetic_question(
    derived_gold_chunk_ids: list[str] | None = None,
    quote_matches: list[list[str]] | None = None,
) -> Question:
    quote_matches = quote_matches or [
        ["act01_scene01_chunk002"],
        ["act01_scene02_chunk001"],
    ]
    evidence_quotes = [
        RequiredEvidenceQuote(
            act=1,
            scene=1,
            speaker="SPEAKER",
            quote="gamma delta epsilon",
            role="answer",
            matched_chunk_ids=quote_matches[0],
        ),
        RequiredEvidenceQuote(
            act=1,
            scene=2,
            speaker="SPEAKER",
            quote="zeta eta",
            role="bridge",
            matched_chunk_ids=quote_matches[1],
        ),
    ]
    return Question(
        id="q",
        question="What happened?",
        expected_answer="Something.",
        evidence_scope="synthetic",
        reasoning_skill="local_fact",
        required_evidence_quotes=evidence_quotes,
        derived_gold_chunk_ids=derived_gold_chunk_ids
        or ["act01_scene01_chunk002", "act01_scene02_chunk001"],
        notes="test",
    )


class SelectionAndPromptTests(unittest.TestCase):
    def setUp(self):
        self.chunks = synthetic_chunks()
        self.lookup = chunks_by_id(self.chunks)
        self.doc_order = document_order_chunk_ids(self.chunks)
        self.global_index_lookup = {
            int(chunk["global_index"]): str(chunk["chunk_id"])
            for chunk in self.chunks
        }
        self.question = synthetic_question()
        self.trace = [
            {"rank": 1, "chunk_id": "act01_scene02_chunk001", "score": 0.9},
            {"rank": 2, "chunk_id": "act01_scene01_chunk002", "score": 0.8},
            {"rank": 3, "chunk_id": "act01_scene01_chunk001", "score": 0.7},
        ]

    def test_evidence_quote_recall_counts_required_quotes_present(self):
        quote_presence = required_quotes_present_in_context(
            self.question,
            ["act01_scene01_chunk002"],
        )

        self.assertEqual(evidence_quote_recall(quote_presence), 0.5)
        self.assertEqual(
            [row["present"] for row in quote_presence],
            [True, False],
        )

    def test_closed_book_contains_no_context(self):
        prepared = prepare_treatment(
            self.question,
            "closed_book",
            5,
            self.lookup,
            self.doc_order,
        )

        self.assertEqual(prepared["selected_chunk_ids"], [])
        self.assertEqual(prepared["context_tokens"], 0)
        self.assertEqual(prepared["evidence_chunk_recall"], 0.0)
        self.assertEqual(prepared["evidence_quote_recall"], 0.0)
        self.assertEqual(prepared["retrieval_scores"], [])
        self.assertEqual(prepared["prompt_order"], "none")

        row = build_result_row(
            self.question,
            "closed_book",
            5,
            prepared,
            StubReader(),
            RunConfig(run_name="unit", prepare_only=True),
            model_output=None,
        )
        self.assertNotIn("Context chunks:", row["user_prompt"])
        self.assertIn("No document context is provided", row["user_prompt"])

    def test_gold_evidence_includes_derived_gold_chunks_by_relevance(self):
        prepared = prepare_treatment(
            self.question,
            "gold_evidence",
            5,
            self.lookup,
            self.doc_order,
            dense_trace=self.trace,
        )

        self.assertEqual(
            prepared["selected_chunk_ids"],
            ["act01_scene02_chunk001", "act01_scene01_chunk002"],
        )
        self.assertEqual(prepared["context_tokens"], 5)
        self.assertEqual(prepared["evidence_chunk_recall"], 1.0)
        self.assertEqual(prepared["evidence_quote_recall"], 1.0)
        self.assertEqual(prepared["prompt_order"], "gold_chunks_by_relevance")

    def test_gold_evidence_neighbors_adds_previous_and_next_chunks(self):
        question = synthetic_question(
            derived_gold_chunk_ids=["act01_scene01_chunk002"],
            quote_matches=[["act01_scene01_chunk002"], ["act01_scene01_chunk002"]],
        )
        trace = [{"rank": 1, "chunk_id": "act01_scene01_chunk002", "score": 0.95}]

        prepared = prepare_treatment(
            question,
            "gold_evidence_neighbors",
            8,
            self.lookup,
            self.doc_order,
            global_index_lookup=self.global_index_lookup,
            dense_trace=trace,
            neighbor_window=1,
        )

        self.assertEqual(
            set(prepared["selected_chunk_ids"]),
            {
                "act01_scene01_chunk001",
                "act01_scene01_chunk002",
                "act01_scene02_chunk001",
            },
        )
        self.assertEqual(
            set(prepared["neighbor_chunk_ids_added"]),
            {"act01_scene01_chunk001", "act01_scene02_chunk001"},
        )
        self.assertEqual(
            prepared["original_hit_chunk_ids"],
            ["act01_scene01_chunk002"],
        )

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
        self.assertEqual(prepared["evidence_chunk_recall"], 0.5)
        self.assertEqual(
            prepared["retrieval_scores"],
            [
                {"chunk_id": "act01_scene02_chunk001", "rank": 1, "score": 0.9},
                {"chunk_id": "act01_scene01_chunk001", "rank": 2, "score": 0.8},
            ],
        )
        self.assertEqual(prepared["prompt_order"], "retrieval_score")

    def test_dense_relevance_neighbors_adds_previous_and_next_chunks_around_hits(self):
        trace = [{"rank": 1, "chunk_id": "act01_scene01_chunk002", "score": 0.95}]

        prepared = prepare_treatment(
            self.question,
            "dense_relevance_neighbors",
            8,
            self.lookup,
            self.doc_order,
            global_index_lookup=self.global_index_lookup,
            dense_trace=trace,
            neighbor_window=1,
        )

        self.assertEqual(
            prepared["selected_chunk_ids"],
            [
                "act01_scene01_chunk001",
                "act01_scene01_chunk002",
                "act01_scene02_chunk001",
            ],
        )
        self.assertEqual(
            set(prepared["neighbor_chunk_ids_added"]),
            {"act01_scene01_chunk001", "act01_scene02_chunk001"},
        )
        self.assertEqual(prepared["original_hit_chunk_ids"], ["act01_scene01_chunk002"])
        self.assertEqual(prepared["prompt_order"], "retrieval_rank_local_neighbor_blocks")

    def test_all_treatments_enforce_context_budget(self):
        for treatment in [
            "closed_book",
            "gold_evidence",
            "gold_evidence_neighbors",
            "dense_relevance",
            "dense_relevance_neighbors",
        ]:
            with self.subTest(treatment=treatment):
                prepared = prepare_treatment(
                    self.question,
                    treatment,
                    5,
                    self.lookup,
                    self.doc_order,
                    global_index_lookup=self.global_index_lookup,
                    dense_trace=self.trace,
                    neighbor_window=1,
                )

                self.assertLessEqual(prepared["context_tokens"], 5)

    def test_grounded_prompt_logs_context_and_asks_for_evidence(self):
        prepared = prepare_treatment(
            self.question,
            "gold_evidence",
            10,
            self.lookup,
            self.doc_order,
            dense_trace=self.trace,
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
        self.assertIn("Context chunks:", row["user_prompt"])
        self.assertIn("using only the provided context", row["user_prompt"])
        self.assertIn("evidence", row["user_prompt"])
        self.assertIn("SYSTEM:", row["full_prompt"])
        self.assertIsNone(row["model_output"])
        self.assertEqual(row["model_name"], config.reader_model)
        self.assertEqual(row["model_max_context"], 4096)
        self.assertEqual(row["max_new_tokens"], config.max_new_tokens)
        self.assertGreater(row["prompt_tokens"], row["context_tokens"])
        self.assertEqual(row["total_input_tokens"], row["prompt_tokens"])
        self.assertIsNone(row["failure_label"])
        self.assertIn(row["raw_chunks"][0]["chunk_id"], row["selected_chunk_ids"])


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
