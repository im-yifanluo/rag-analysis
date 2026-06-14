from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hamlet_qa.core.config import BASELINE_TREATMENTS, RunConfig
from hamlet_qa.core.questions import Question, RequiredEvidenceQuote
from hamlet_qa.core.experiment import (
    build_result_row,
    chunks_by_id,
    document_order_chunk_ids,
    evidence_quote_recall,
    prepare_treatment,
    required_quotes_present_in_context,
    run_experiment,
)


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


class RecordingReader:
    def __init__(self, events: list[str]):
        self.events = events
        self.model_max_context = 4096

    def format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        return f"SYSTEM: {system_prompt}\nUSER: {user_prompt}\nASSISTANT:"

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        del max_tokens
        del system_prompt, user_prompt
        self.events.append("generate")
        return "answer"


class SetRPrepareOnlyReader(StubReader):
    model_name = "fake-reader"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        self.generate_calls += 1
        self.last_selector_prompt = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "max_tokens": max_tokens,
        }
        return "Step 1. Need selected evidence.\n### Final Selection: [1]"


class RecordingRetriever:
    def __init__(self, events: list[str]):
        self.events = events

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        del query, top_k
        self.events.append("retrieve")
        return [
            {
                "rank": 1,
                "chunk_id": "act01_scene01_chunk001",
                "score": 1.0,
                "dense_rank": 1,
                "dense_score": 1.0,
                "rerank_score": 1.0,
                "retrieval_method": "dense_faiss_reranked",
                "global_index": 0,
                "act": 1,
                "scene": 1,
                "scene_title": "First.",
            }
        ]


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

    def test_gold_evidence_includes_derived_gold_chunks_by_document_order(self):
        prepared = prepare_treatment(
            self.question,
            "gold_evidence",
            5,
            self.lookup,
            self.doc_order,
        )

        self.assertEqual(
            prepared["selected_chunk_ids"],
            ["act01_scene01_chunk002", "act01_scene02_chunk001"],
        )
        self.assertEqual(prepared["context_tokens"], 5)
        self.assertEqual(prepared["evidence_chunk_recall"], 1.0)
        self.assertEqual(prepared["evidence_quote_recall"], 1.0)
        self.assertEqual(prepared["prompt_order"], "gold_chunks_document_order")
        self.assertEqual(prepared["retrieval_method"], "gold")

    def test_gold_evidence_covers_every_quote_before_filling_budget(self):
        # Regression for q_state_poisoned_cup: overlapping chunks satisfy the
        # same quote, and greedy budget filling used to exhaust the budget on
        # redundant chunks before reaching the only chunk holding the last
        # quote (4 x 256-token gold chunks vs a 1000-token budget).
        chunks = [
            {
                "chunk_id": f"act05_scene02_chunk{i:03d}",
                "global_index": i,
                "act": 5,
                "scene": 2,
                "scene_id": "act05_scene02",
                "scene_title": "Final.",
                "chunk_in_scene": i,
                "start_token": i * 192,
                "end_token": i * 192 + 256,
                "token_count": 256,
                "text": f"text {i}",
            }
            for i in range(19, 23)
        ]
        lookup = chunks_by_id(chunks)
        doc_order = document_order_chunk_ids(chunks)
        question = Question(
            id="q_cup",
            question="Who drinks?",
            expected_answer="Gertrude.",
            evidence_scope="Act 5, Scene 2",
            reasoning_skill="entity_state_tracking",
            required_evidence_quotes=[
                RequiredEvidenceQuote(
                    act=5,
                    scene=2,
                    speaker="KING",
                    quote="cup setup",
                    role="cup_setup",
                    matched_chunk_ids=["act05_scene02_chunk019"],
                ),
                RequiredEvidenceQuote(
                    act=5,
                    scene=2,
                    speaker="KING",
                    quote="cup state",
                    role="cup_state",
                    matched_chunk_ids=[
                        "act05_scene02_chunk020",
                        "act05_scene02_chunk021",
                    ],
                ),
                RequiredEvidenceQuote(
                    act=5,
                    scene=2,
                    speaker="QUEEN",
                    quote="outcome",
                    role="outcome",
                    matched_chunk_ids=["act05_scene02_chunk022"],
                ),
            ],
            derived_gold_chunk_ids=[chunk["chunk_id"] for chunk in chunks],
            notes="regression",
        )

        prepared = prepare_treatment(
            question,
            "gold_evidence",
            1000,
            lookup,
            doc_order,
        )

        self.assertEqual(prepared["evidence_quote_recall"], 1.0)
        self.assertIn("act05_scene02_chunk022", prepared["selected_chunk_ids"])
        self.assertLessEqual(prepared["context_tokens"], 1000)

    def test_quote_presence_detects_evidence_text_inside_pseudo_chunks(self):
        quote_presence = required_quotes_present_in_context(
            self.question,
            ["synthetic_summary"],
            selected_chunks=[
                {
                    "chunk_id": "synthetic_summary",
                    "token_count": 6,
                    "text": "Summary: Gamma Delta  Epsilon happened first.",
                }
            ],
        )

        self.assertEqual(
            [row["present"] for row in quote_presence],
            [True, False],
        )
        self.assertFalse(quote_presence[0]["present_by_chunk_id"])
        self.assertTrue(quote_presence[0]["present_by_text"])

    def test_dense_reranked_uses_reranker_order_and_logs_scores(self):
        trace = [
            {
                "rank": 1,
                "chunk_id": "act01_scene02_chunk001",
                "score": 0.9,
                "dense_rank": 2,
                "dense_score": 0.3,
                "rerank_score": 0.9,
                "retrieval_method": "dense_faiss_reranked",
            },
            {
                "rank": 2,
                "chunk_id": "act01_scene01_chunk001",
                "score": 0.8,
                "dense_rank": 1,
                "dense_score": 0.7,
                "rerank_score": 0.8,
                "retrieval_method": "dense_faiss_reranked",
            },
            {
                "rank": 3,
                "chunk_id": "act01_scene01_chunk002",
                "score": 0.7,
                "dense_rank": 3,
                "dense_score": 0.1,
                "rerank_score": 0.7,
                "retrieval_method": "dense_faiss_reranked",
            },
        ]
        prepared = prepare_treatment(
            self.question,
            "dense_reranked",
            5,
            self.lookup,
            self.doc_order,
            retrieval_trace=trace,
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
                {
                    "chunk_id": "act01_scene02_chunk001",
                    "rank": 1,
                    "score": 0.9,
                    "dense_rank": 2,
                    "dense_score": 0.3,
                    "rerank_score": 0.9,
                    "retrieval_method": "dense_faiss_reranked",
                },
                {
                    "chunk_id": "act01_scene01_chunk001",
                    "rank": 2,
                    "score": 0.8,
                    "dense_rank": 1,
                    "dense_score": 0.7,
                    "rerank_score": 0.8,
                    "retrieval_method": "dense_faiss_reranked",
                },
            ],
        )
        self.assertEqual(prepared["prompt_order"], "dense_reranker_rank")
        self.assertEqual(prepared["retrieval_method"], "dense_faiss_reranked")

    def test_dense_document_order_uses_same_hits_in_document_order(self):
        prepared = prepare_treatment(
            self.question,
            "dense_document_order",
            8,
            self.lookup,
            self.doc_order,
            retrieval_trace=self.trace,
        )

        self.assertEqual(
            prepared["selected_chunk_ids"],
            [
                "act01_scene01_chunk001",
                "act01_scene01_chunk002",
                "act01_scene02_chunk001",
            ],
        )
        self.assertEqual(prepared["original_hit_chunk_ids"], [
            "act01_scene02_chunk001",
            "act01_scene01_chunk002",
            "act01_scene01_chunk001",
        ])
        self.assertEqual(prepared["prompt_order"], "dense_hits_document_order")

    def test_dense_random_order_uses_same_hits_with_seeded_shuffle(self):
        prepared_a = prepare_treatment(
            self.question,
            "dense_random_order",
            8,
            self.lookup,
            self.doc_order,
            retrieval_trace=self.trace,
            random_seed=99,
        )
        prepared_b = prepare_treatment(
            self.question,
            "dense_random_order",
            8,
            self.lookup,
            self.doc_order,
            retrieval_trace=self.trace,
            random_seed=99,
        )

        self.assertEqual(prepared_a["selected_chunk_ids"], prepared_b["selected_chunk_ids"])
        self.assertEqual(
            set(prepared_a["selected_chunk_ids"]),
            {"act01_scene01_chunk001", "act01_scene01_chunk002", "act01_scene02_chunk001"},
        )
        self.assertEqual(prepared_a["prompt_order"], "dense_hits_random_order")

    def test_all_treatments_enforce_context_budget(self):
        for treatment in [
            "closed_book",
            "gold_evidence",
            "dense_reranked",
            "dense_document_order",
            "dense_random_order",
            "sparse_bm25",
        ]:
            with self.subTest(treatment=treatment):
                trace = self.trace if treatment != "sparse_bm25" else [
                    {"rank": 1, "chunk_id": "act01_scene01_chunk001", "score": 2.0},
                    {"rank": 2, "chunk_id": "act01_scene02_chunk001", "score": 1.5},
                    {"rank": 3, "chunk_id": "act01_scene01_chunk002", "score": 1.0},
                ]
                prepared = prepare_treatment(
                    self.question,
                    treatment,
                    5,
                    self.lookup,
                    self.doc_order,
                    retrieval_trace=trace,
                )

                self.assertLessEqual(prepared["context_tokens"], 5)

    def test_grounded_prompt_logs_context_and_asks_for_evidence(self):
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
        self.assertIn("Context chunks:", row["user_prompt"])
        self.assertIn("using only the provided context", row["user_prompt"])
        self.assertIn("evidence", row["user_prompt"])
        self.assertIn("SYSTEM:", row["full_prompt"])
        self.assertIsNone(row["model_output"])
        self.assertEqual(row["model_name"], config.reader_model)
        self.assertEqual(row["embedding_model"], config.embedding_model)
        self.assertEqual(row["reranker_model"], config.reranker_model)
        self.assertEqual(row["gpu_layout"], config.gpu_layout)
        self.assertEqual(row["embedding_device"], config.embedding_device)
        self.assertEqual(row["reranker_device"], config.reranker_device)
        self.assertEqual(row["reader_device"], config.reader_device)
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
            self.assertEqual(len(rows), 10)
            self.assertTrue(all(row["model_output"] is None for row in rows))
            self.assertTrue(all(row["selected_chunk_ids"] == [] for row in rows))

    def test_dense_retrieval_is_precomputed_before_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            events: list[str] = []
            config = RunConfig(
                chunks_path=str(REPO_ROOT / "data" / "hamlet_chunks.jsonl"),
                questions_path=str(REPO_ROOT / "data" / "hamlet_questions.json"),
                output_dir=tmp,
                run_name="precompute",
                treatments=["dense_reranked"],
                context_budgets=[500],
                prepare_only=False,
            )

            run_experiment(
                config,
                reader=RecordingReader(events),
                dense_retriever=RecordingRetriever(events),
            )

            first_generate = events.index("generate")
            self.assertTrue(events[:first_generate])
            self.assertTrue(all(event == "retrieve" for event in events[:first_generate]))

    def test_existing_treatments_still_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            events: list[str] = []
            config = RunConfig(
                chunks_path=str(REPO_ROOT / "data" / "hamlet_chunks.jsonl"),
                questions_path=str(REPO_ROOT / "data" / "hamlet_questions.json"),
                output_dir=tmp,
                run_name="baseline_smoke",
                treatments=BASELINE_TREATMENTS.copy(),
                context_budgets=[300],
                prepare_only=True,
            )

            results_path = run_experiment(
                config,
                reader=StubReader(),
                dense_retriever=RecordingRetriever(events),
            )

            rows = [
                json.loads(line)
                for line in Path(results_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 10 * len(BASELINE_TREATMENTS))
            self.assertEqual({row["treatment"] for row in rows}, set(BASELINE_TREATMENTS))
            self.assertTrue(all(row["model_output"] is None for row in rows))

    def test_prepare_only_works_for_new_context_assembly_treatments(self):
        with tempfile.TemporaryDirectory() as tmp:
            events: list[str] = []
            config = RunConfig(
                chunks_path=str(REPO_ROOT / "data" / "hamlet_chunks.jsonl"),
                questions_path=str(REPO_ROOT / "data" / "hamlet_questions.json"),
                output_dir=tmp,
                run_name="new_prepare_only",
                treatments=["setr", "domain"],
                context_budgets=[300],
                prepare_only=True,
                context_assembly_cache_dir=str(Path(tmp) / "cache"),
            )

            reader = SetRPrepareOnlyReader()
            results_path = run_experiment(
                config,
                reader=reader,
                dense_retriever=RecordingRetriever(events),
            )

            rows = [
                json.loads(line)
                for line in Path(results_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 20)
            self.assertEqual({row["treatment"] for row in rows}, {"setr", "domain"})
            self.assertTrue(all(row["model_output"] is None for row in rows))
            self.assertTrue(all(row["context_assembly_trace"] for row in rows))
            self.assertEqual(reader.generate_calls, 10)
            setr_rows = [row for row in rows if row["treatment"] == "setr"]
            self.assertTrue(
                all(
                    row["context_assembly_trace"]["selector_model"] == "fake-reader"
                    for row in setr_rows
                )
            )
            domain_rows = [row for row in rows if row["treatment"] == "domain"]
            self.assertTrue(
                all(row["selected_chunk_ids"][0] == "domain_scaffold" for row in domain_rows)
            )

    def test_prepare_only_works_for_method_treatments_with_stub_handles(self):
        class StubEvaluator:
            def score(self, query: str, documents: list[str]) -> list[float]:
                del query
                return [5.0 for _document in documents]

        class StubMacragRetriever:
            def __init__(self, chunks: list[dict]):
                self.chunk = chunks[0]

            def retrieve(self, query: str, top_k: int) -> list[dict]:
                del query, top_k
                return [
                    {
                        "chunk_id": self.chunk["chunk_id"],
                        "rank": 1,
                        "score": 1.0,
                        "rerank_score": 1.0,
                        "slice_id": f"{self.chunk['chunk_id']}_summary00",
                        "slice_rank": 1,
                        "retrieval_method": "macrag_summary_slices_reranked",
                    }
                ]

        with tempfile.TemporaryDirectory() as tmp:
            events: list[str] = []
            chunks = [
                json.loads(line)
                for line in (REPO_ROOT / "data" / "hamlet_chunks.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            config = RunConfig(
                chunks_path=str(REPO_ROOT / "data" / "hamlet_chunks.jsonl"),
                questions_path=str(REPO_ROOT / "data" / "hamlet_questions.json"),
                output_dir=tmp,
                run_name="methods_prepare_only",
                treatments=["crag", "macrag", "recomp_extractive"],
                # 500: a macrag chunk_ext=1 merged block of two 256-token
                # chunks with 64-token overlap is 448 tokens.
                context_budgets=[500],
                prepare_only=True,
                context_assembly_cache_dir=str(Path(tmp) / "cache"),
            )
            questions = [
                json.loads(line)["question_id"]
                for line in []
            ]
            del questions
            question_ids = [
                row["id"]
                for row in json.loads(
                    (REPO_ROOT / "data" / "hamlet_questions.json").read_text(
                        encoding="utf-8"
                    )
                )
            ]
            handles = {
                "crag_evaluator": StubEvaluator(),
                "crag_reretriever": RecordingRetriever(events),
                "macrag_retriever": StubMacragRetriever(chunks),
                "recomp_summaries": {
                    f"recomp_extractive:{question_id}": {
                        "summary": "Stub compressed evidence summary.",
                        "compressor_model": "stub",
                    }
                    for question_id in question_ids
                },
            }

            reader = SetRPrepareOnlyReader()
            results_path = run_experiment(
                config,
                reader=reader,
                dense_retriever=RecordingRetriever(events),
                feature_handles=handles,
            )

            rows = [
                json.loads(line)
                for line in Path(results_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 30)
            by_treatment: dict[str, list[dict]] = {}
            for row in rows:
                by_treatment.setdefault(row["treatment"], []).append(row)
            self.assertEqual(
                set(by_treatment),
                {"crag", "macrag", "recomp_extractive"},
            )
            self.assertTrue(
                all(
                    row["selected_chunk_ids"] == ["crag_refined_knowledge"]
                    for row in by_treatment["crag"]
                )
            )
            # rerank_score 1.0 sits between the calibrated thresholds, so the
            # ambiguous action fires and exercises rewrite + re-retrieval.
            self.assertTrue(
                all(
                    row["context_assembly_trace"]["action"] == "ambiguous"
                    for row in by_treatment["crag"]
                )
            )
            self.assertTrue(
                all(
                    "external_knowledge" in row["context_assembly_trace"]
                    for row in by_treatment["crag"]
                )
            )
            self.assertTrue(
                all(
                    row["selected_chunk_ids"] == ["recomp_extractive_summary"]
                    for row in by_treatment["recomp_extractive"]
                )
            )
            self.assertTrue(
                all(
                    row["retrieval_method"] == "macrag_summary_slices_reranked"
                    for row in by_treatment["macrag"]
                )
            )
            self.assertTrue(
                all(row["selected_chunk_ids"] for row in by_treatment["macrag"])
            )


if __name__ == "__main__":
    unittest.main()
