from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class HygieneTests(unittest.TestCase):
    def test_hamlet_package_has_no_old_benchmark_references(self):
        banned = [
            "benchmarking",
            "SCROLLS",
            "RAPTOR",
            "ReadAgent",
            "DOS-RAG",
            "dos_rag",
            "read_agent",
        ]
        for path in (REPO_ROOT / "hamlet_qa").glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for term in banned:
                with self.subTest(path=path.name, term=term):
                    self.assertNotIn(term, text)


if __name__ == "__main__":
    unittest.main()
