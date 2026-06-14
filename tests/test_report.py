from __future__ import annotations

import unittest

from hamlet_qa.inspection.report import render_inspection_report


class InspectionReportTests(unittest.TestCase):
    def test_inspection_report_renders(self):
        rows = [
            {
                "question_id": "q1",
                "treatment": "closed_book",
                "selected_chunk_ids": [],
                "context_tokens": 0,
                "evidence_chunk_recall": 0.0,
                "evidence_quote_recall": 0.0,
                "model_output": "A concise answer.",
            },
            {
                "question_id": "q1",
                "treatment": "gold_evidence",
                "selected_chunk_ids": ["act01_scene01_chunk001"],
                "context_tokens": 256,
                "evidence_chunk_recall": 1.0,
                "evidence_quote_recall": 1.0,
                "model_output": None,
            },
        ]

        report = render_inspection_report(rows)

        self.assertIn("# Hamlet QA Inspection Report", report)
        self.assertIn("## Treatment Summary", report)
        self.assertIn("gold_evidence", report)
        self.assertIn("act01_scene01_chunk001", report)
        # Metric columns render as n/a when the sidecar is absent.
        self.assertIn("sufficient context rate", report)
        self.assertIn("mean CI+ fraction", report)

    def test_inspection_report_renders_metric_annotations_when_present(self):
        rows = [
            {
                "question_id": "q1",
                "treatment": "dense_reranked",
                "selected_chunk_ids": ["c1"],
                "context_tokens": 256,
                "evidence_chunk_recall": 1.0,
                "evidence_quote_recall": 1.0,
                "model_output": "Answer.",
                "sufficient_context": 1,
                "ci_positive_fraction": 0.5,
            },
        ]

        report = render_inspection_report(rows)

        summary_line = next(
            line for line in report.splitlines() if line.startswith("| dense_reranked")
        )
        self.assertIn("1.000", summary_line)
        self.assertIn("0.500", summary_line)


if __name__ == "__main__":
    unittest.main()
