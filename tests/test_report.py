from __future__ import annotations

import unittest

from hamlet_qa.report import render_inspection_report


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


if __name__ == "__main__":
    unittest.main()
