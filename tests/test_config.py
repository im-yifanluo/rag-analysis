from __future__ import annotations

import unittest
from unittest.mock import patch

from hamlet_qa.core.config import (
    BASELINE_TREATMENTS,
    DEFAULT_CONTEXT_ASSEMBLY_CACHE_DIR,
    DEFAULT_DOMAIN_KG_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GPU_LAYOUT,
    DEFAULT_READER_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_SETR_MAX_PASSAGES,
    DEFAULT_SETR_SELECTOR_MAX_TOKENS,
    DEFAULT_TREATMENTS,
    DEFAULT_TOKENIZER_MODEL,
    DEFAULT_TOP_K,
    GPU_LAYOUTS,
    ORDERING_TREATMENTS,
    RunConfig,
)
from hamlet_qa.cli.run_experiment import config_from_args, parse_args


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
        self.assertEqual(DEFAULT_GPU_LAYOUT, "single")
        self.assertEqual(config.gpu_layout, "single")
        self.assertEqual(config.embedding_device, "cuda")
        self.assertEqual(config.reranker_device, "cuda")
        self.assertEqual(config.reader_device, "cuda")
        self.assertEqual(
            GPU_LAYOUTS["a40-2gpu"],
            {
                "embedding_device": "cuda:0",
                "reranker_device": "cuda:0",
                "reader_device": "cuda:1",
            },
        )
        self.assertEqual(
            GPU_LAYOUTS["a40-3gpu"],
            {
                "embedding_device": "cuda:0",
                "reranker_device": "cuda:1",
                "reader_device": "cuda:2",
            },
        )
        self.assertEqual(
            BASELINE_TREATMENTS,
            [
                "closed_book",
                "gold_evidence",
                "dense_reranked",
                "sparse_bm25",
            ],
        )
        self.assertEqual(
            ORDERING_TREATMENTS,
            [
                "dense_document_order",
                "dense_random_order",
            ],
        )
        self.assertEqual(
            DEFAULT_TREATMENTS,
            BASELINE_TREATMENTS
            + ORDERING_TREATMENTS
            + ["setr", "domain"]
            + ["crag", "macrag", "recomp_extractive", "recomp_abstractive"]
            + ["reader_support"],
        )
        self.assertEqual(config.domain_kg_path, DEFAULT_DOMAIN_KG_PATH)
        self.assertEqual(config.context_assembly_cache_dir, DEFAULT_CONTEXT_ASSEMBLY_CACHE_DIR)
        self.assertEqual(DEFAULT_SETR_MAX_PASSAGES, DEFAULT_TOP_K)
        self.assertEqual(config.setr_max_passages, DEFAULT_SETR_MAX_PASSAGES)
        self.assertEqual(config.setr_selector_max_tokens, DEFAULT_SETR_SELECTOR_MAX_TOKENS)

    def test_cli_a40_3gpu_layout_sets_expected_devices(self):
        with patch("sys.argv", ["run_experiment", "--gpu-layout", "a40-3gpu"]):
            config = config_from_args(parse_args())

        self.assertEqual(config.gpu_layout, "a40-3gpu")
        self.assertEqual(config.embedding_device, "cuda:0")
        self.assertEqual(config.reranker_device, "cuda:1")
        self.assertEqual(config.reader_device, "cuda:2")

    def test_cli_a40_2gpu_layout_sets_expected_devices(self):
        with patch("sys.argv", ["run_experiment", "--gpu-layout", "a40-2gpu"]):
            config = config_from_args(parse_args())

        self.assertEqual(config.gpu_layout, "a40-2gpu")
        self.assertEqual(config.embedding_device, "cuda:0")
        self.assertEqual(config.reranker_device, "cuda:0")
        self.assertEqual(config.reader_device, "cuda:1")

    def test_cli_device_overrides_take_precedence_over_layout(self):
        with patch(
            "sys.argv",
            [
                "run_experiment",
                "--gpu-layout",
                "a40-3gpu",
                "--embedding-device",
                "cuda:2",
            ],
        ):
            config = config_from_args(parse_args())

        self.assertEqual(config.embedding_device, "cuda:2")
        self.assertEqual(config.reranker_device, "cuda:1")
        self.assertEqual(config.reader_device, "cuda:2")


if __name__ == "__main__":
    unittest.main()
