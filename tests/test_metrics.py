from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hamlet_qa.metrics.annotate import (
    annotate_results,
    annotations_path_for,
    load_annotations,
    merge_annotations_into_rows,
)
from hamlet_qa.metrics.ci import compute_ci_for_row
from hamlet_qa.metrics.sufficient_context import (
    SUFFICIENT_CONTEXT_PROMPT_TEMPLATE,
    build_sufficient_context_prompt,
    compute_sufficient_context_for_row,
    context_text_for_row,
    parse_autorater_output,
)


def result_row(chunks: list[dict] | None = None) -> dict:
    return {
        "question_id": "q1",
        "treatment": "dense_reranked",
        "context_budget": 1000,
        "question": "Who drinks the poison?",
        "expected_answer": "Gertrude drinks the poison.",
        "raw_chunks": chunks
        if chunks is not None
        else [
            {
                "chunk_id": "c_good",
                "act": 5,
                "scene": 2,
                "scene_title": "Final.",
                "token_count": 8,
                "text": "The Queen drinks and cries the drink the drink.",
            },
            {
                "chunk_id": "c_noise",
                "act": 1,
                "scene": 1,
                "scene_title": "First.",
                "token_count": 6,
                "text": "Barnardo asks who is there.",
            },
        ],
    }


class ScriptedScorer:
    """Maps frozenset of included chunk_ids -> mean logprob of the answer."""

    model_name = "scripted"

    def __init__(self, logprob_by_chunks: dict[frozenset, float]):
        self.logprob_by_chunks = logprob_by_chunks
        self.last_chunks: list[list[str]] = []

    def format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        chunk_ids = []
        for line in user_prompt.splitlines():
            if line.startswith("[") and "|" in line:
                chunk_ids.append(line[1:].split(" |")[0].strip())
        self.last_chunks.append(chunk_ids)
        return "||".join(sorted(chunk_ids))

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def score_completion(self, full_prompt: str, completion: str) -> dict:
        del completion
        chunk_ids = frozenset(part for part in full_prompt.split("||") if part)
        mean_logprob = self.logprob_by_chunks[chunk_ids]
        return {
            "sum_logprob": mean_logprob * 4,
            "num_tokens": 4,
            "mean_logprob": mean_logprob,
        }


class CiMetricTests(unittest.TestCase):
    def test_ci_signs_follow_leave_one_out_loss_change(self):
        # Full context scores best; dropping c_good hurts (phi > 0), dropping
        # c_noise helps slightly (phi < 0).
        scorer = ScriptedScorer(
            {
                frozenset({"c_good", "c_noise"}): -1.0,
                frozenset({"c_noise"}): -3.0,
                frozenset({"c_good"}): -0.8,
            }
        )
        annotation = compute_ci_for_row(result_row(), scorer)

        self.assertAlmostEqual(annotation["ci_base_loss"], 1.0)
        by_chunk = {item["chunk_id"]: item for item in annotation["ci_values"]}
        self.assertAlmostEqual(by_chunk["c_good"]["phi"], 2.0)
        self.assertAlmostEqual(by_chunk["c_noise"]["phi"], -0.2)
        self.assertEqual(annotation["ci_positive_chunk_ids"], ["c_good"])
        self.assertEqual(annotation["ci_positive_fraction"], 0.5)

    def test_ci_skips_rows_without_context(self):
        annotation = compute_ci_for_row(result_row(chunks=[]), ScriptedScorer({}))

        self.assertIsNone(annotation["ci_base_loss"])
        self.assertIsNone(annotation["ci_values"])
        self.assertIsNone(annotation["ci_positive_fraction"])


class FakeRater:
    model_name = "fake-rater"

    def __init__(self, output: str):
        self.output = output
        self.prompts: list[str] = []

    def generate(self, system_prompt: str, user_prompt: str, max_tokens=None) -> str:
        del system_prompt, max_tokens
        self.prompts.append(user_prompt)
        return self.output


class SufficientContextTests(unittest.TestCase):
    def test_prompt_contains_one_shot_example_and_row_content(self):
        prompt = build_sufficient_context_prompt("Who?", "Some context.")

        self.assertIn("Roald Dahl's Guide to Railway Safety", prompt)
        self.assertIn('{"Sufficient Context": 1}', prompt)
        self.assertIn("### QUESTION\nWho?", prompt)
        self.assertIn("### REFERENCES\nSome context.", prompt)
        # The deviation: no timestamp placeholder remains.
        self.assertNotIn("TIMESTAMP", SUFFICIENT_CONTEXT_PROMPT_TEMPLATE)

    def test_parse_label_and_explanation(self):
        parsed = parse_autorater_output(
            "### EXPLANATION\nThe context names the drinker.\n"
            '### JSON\n{"Sufficient Context": 1}'
        )
        self.assertEqual(parsed["label"], 1)
        self.assertIn("names the drinker", parsed["explanation"])

        fallback = parse_autorater_output("Sufficient Context: 0")
        self.assertEqual(fallback["label"], 0)

        unparseable = parse_autorater_output("no label here")
        self.assertIsNone(unparseable["label"])

    def test_compute_for_row_uses_chunk_context_and_parses_label(self):
        rater = FakeRater('### EXPLANATION\nFine.\n### JSON\n{"Sufficient Context": 0}')
        annotation = compute_sufficient_context_for_row(result_row(), rater)

        self.assertEqual(annotation["sufficient_context"], 0)
        self.assertIn("The Queen drinks", rater.prompts[0])
        self.assertIn("[c_good | Act 5 Scene 2", rater.prompts[0])

    def test_closed_book_rows_use_no_context_placeholder(self):
        self.assertEqual(context_text_for_row(result_row(chunks=[])), "[no context]")


class AnnotationSidecarTests(unittest.TestCase):
    def test_annotate_results_writes_sidecar_and_merges_on_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_path = Path(tmp) / "results.jsonl"
            rows = [result_row(), dict(result_row(), treatment="closed_book")]
            with results_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            annotate_results(
                results_path,
                {"sufficient_context": lambda row: {"sufficient_context": 1}},
            )

            self.assertTrue(annotations_path_for(results_path).exists())
            annotations = load_annotations(results_path)
            self.assertEqual(len(annotations), 2)
            merged = merge_annotations_into_rows(rows, annotations)
            self.assertTrue(all(row["sufficient_context"] == 1 for row in merged))

    def test_annotate_results_is_idempotent_unless_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_path = Path(tmp) / "results.jsonl"
            with results_path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(result_row()) + "\n")

            calls: list[str] = []

            def metric(row: dict) -> dict:
                calls.append(row["question_id"])
                return {"sufficient_context": 1}

            annotate_results(results_path, {"sufficient_context": metric})
            annotate_results(results_path, {"sufficient_context": metric})
            self.assertEqual(len(calls), 1)

            annotate_results(
                results_path,
                {"sufficient_context": metric},
                overwrite=True,
            )
            self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
