from __future__ import annotations

import unittest

from hamlet_qa.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_READER_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_TOKENIZER_MODEL,
    RunConfig,
)


class ConfigDefaultsTests(unittest.TestCase):
    def test_qwen_defaults_are_aligned(self):
        self.assertEqual(DEFAULT_READER_MODEL, "Qwen/Qwen3.5-9B")
        self.assertEqual(DEFAULT_TOKENIZER_MODEL, DEFAULT_READER_MODEL)
        self.assertEqual(DEFAULT_EMBEDDING_MODEL, "Qwen/Qwen3-Embedding-8B")
        self.assertEqual(DEFAULT_RERANKER_MODEL, "Qwen/Qwen3-Reranker-8B")

        config = RunConfig()

        self.assertEqual(config.tokenizer_model, config.reader_model)
        self.assertEqual(config.embedding_model, DEFAULT_EMBEDDING_MODEL)
        self.assertEqual(config.reranker_model, DEFAULT_RERANKER_MODEL)


if __name__ == "__main__":
    unittest.main()
