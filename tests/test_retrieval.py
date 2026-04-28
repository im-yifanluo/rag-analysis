from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from hamlet_qa.retrieval import DenseRetriever


class FakeFaissIndex:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.embeddings: np.ndarray | None = None

    def add(self, embeddings: np.ndarray) -> None:
        self.embeddings = embeddings

    def search(self, query: np.ndarray, k: int):
        if self.embeddings is None:
            raise AssertionError("index has no embeddings")
        scores = query @ self.embeddings.T
        order = np.argsort(-scores[0])[:k]
        return scores[:, order], order.reshape(1, -1)


class FakeEmbedder:
    def embed_passages(self, passages: list[str]) -> np.ndarray:
        vectors = {
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
            "gamma": [0.8, 0.2],
        }
        return np.array([vectors[passage] for passage in passages], dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        del query
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeReranker:
    def score(self, query: str, documents: list[str]) -> list[float]:
        del query
        scores = {
            "alpha": 0.1,
            "beta": 0.2,
            "gamma": 0.9,
        }
        return [scores[document] for document in documents]


def fake_faiss_module():
    module = types.SimpleNamespace()
    module.IndexFlatIP = FakeFaissIndex
    return module


class DenseRetrieverTests(unittest.TestCase):
    def test_reranker_reorders_dense_candidates_and_preserves_dense_scores(self):
        chunks = [
            {
                "chunk_id": "c_alpha",
                "global_index": 0,
                "act": 1,
                "scene": 1,
                "scene_title": "First.",
                "text": "alpha",
            },
            {
                "chunk_id": "c_beta",
                "global_index": 1,
                "act": 1,
                "scene": 1,
                "scene_title": "First.",
                "text": "beta",
            },
            {
                "chunk_id": "c_gamma",
                "global_index": 2,
                "act": 1,
                "scene": 2,
                "scene_title": "Second.",
                "text": "gamma",
            },
        ]

        with patch.dict(sys.modules, {"faiss": fake_faiss_module()}):
            retriever = DenseRetriever(FakeEmbedder(), chunks, reranker=FakeReranker())
            trace = retriever.retrieve("query", top_k=3)

        self.assertEqual([row["chunk_id"] for row in trace], ["c_gamma", "c_beta", "c_alpha"])
        self.assertEqual([row["rank"] for row in trace], [1, 2, 3])
        self.assertEqual([row["dense_rank"] for row in trace], [2, 3, 1])
        self.assertEqual([row["rerank_score"] for row in trace], [0.9, 0.2, 0.1])
        self.assertEqual(trace[0]["score"], trace[0]["rerank_score"])
        self.assertAlmostEqual(trace[-1]["dense_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
