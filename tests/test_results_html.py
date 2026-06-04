from __future__ import annotations

import unittest

from hamlet_qa.inspection.results_html import render_results_html


class ResultsHtmlTests(unittest.TestCase):
    def test_html_embeds_rows_and_escapes_script_end(self):
        html = render_results_html(
            [
                {
                    "question_id": "q1",
                    "treatment": "gold_evidence",
                    "question": "Who speaks?",
                    "expected_answer": "Francisco.",
                    "model_output": "raw answer </script> still data",
                    "required_quotes_present_in_context": [
                        {
                            "quote": "For this relief much thanks.",
                            "matched_chunk_ids": ["act01_scene01_chunk001"],
                            "present": True,
                        }
                    ],
                    "selected_chunk_ids": ["act01_scene01_chunk001"],
                    "raw_chunks": [
                        {
                            "chunk_id": "act01_scene01_chunk001",
                            "text": "Exact chunk text.",
                        }
                    ],
                }
            ],
            "results.jsonl",
            {
                "act01_scene01_chunk001": {
                    "chunk_id": "act01_scene01_chunk001",
                    "text": "Evidence chunk text from corpus.",
                }
            },
            "hamlet_chunks.jsonl",
        )

        self.assertIn('<script type="application/json" id="embedded-results">', html)
        self.assertIn('<script type="application/json" id="embedded-chunks">', html)
        self.assertIn("raw answer <\\/script> still data", html)
        self.assertIn("Evidence chunk text from corpus.", html)
        self.assertIn("renderEvidenceChunks", html)
        self.assertIn("Raw Prompts", html)
        self.assertIn("Hamlet QA Results", html)


if __name__ == "__main__":
    unittest.main()
