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
        for path in (REPO_ROOT / "hamlet_qa").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for term in banned:
                with self.subTest(path=path.name, term=term):
                    self.assertNotIn(term, text)

    def test_active_source_has_no_retired_model_defaults(self):
        banned = [
            "Qwen" + "2.5",
            "Qwen/" + "Qwen2",
            "Snow" + "flake",
            "snow" + "flake",
            "arc" + "tic",
        ]
        paths = [REPO_ROOT / "README.md", REPO_ROOT / "environment.yml"]
        paths.extend((REPO_ROOT / "hamlet_qa").rglob("*.py"))
        paths.extend((REPO_ROOT / "tests").rglob("*.py"))

        for path in paths:
            text = path.read_text(encoding="utf-8")
            for term in banned:
                with self.subTest(path=path.relative_to(REPO_ROOT), term=term):
                    self.assertNotIn(term, text)


if __name__ == "__main__":
    unittest.main()
