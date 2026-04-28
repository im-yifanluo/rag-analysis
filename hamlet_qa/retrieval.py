"""Dense FAISS retrieval and reranking for the Hamlet QA harness."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Protocol

import numpy as np

from hamlet_qa.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_TOP_K,
)


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


class EmbedderLike(Protocol):
    def embed_passages(self, passages: list[str]) -> np.ndarray:
        ...

    def embed_query(self, query: str) -> np.ndarray:
        ...


class RerankerLike(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]:
        ...


def tokenize_for_bm25(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class SentenceTransformerEmbedder:
    """SentenceTransformers wrapper for the configured Qwen embedding model."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str = "cuda",
        batch_size: int = 64,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True,
        )
        self.batch_size = batch_size

    def embed_passages(self, passages: list[str]) -> np.ndarray:
        return self.model.encode(
            passages,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        kwargs: dict[str, Any] = {
            "show_progress_bar": False,
            "normalize_embeddings": True,
        }
        if "qwen3-embedding" in self.model_name.lower():
            kwargs["prompt_name"] = "query"
        return self.model.encode(query, **kwargs).astype(np.float32)


class CrossEncoderReranker:
    """SentenceTransformers CrossEncoder wrapper for Qwen reranker models."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str = "cuda",
        batch_size: int = 8,
    ):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(
            model_name,
            device=device,
            trust_remote_code=True,
        )

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        pairs = [(query, document) for document in documents]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]


class DenseRetriever:
    """One-document FAISS retriever with an optional cross-encoder reranker."""

    def __init__(
        self,
        embedder: EmbedderLike,
        chunks: list[dict[str, Any]],
        reranker: RerankerLike | None = None,
    ):
        import faiss

        self.embedder = embedder
        self.reranker = reranker
        self.chunks = [dict(chunk) for chunk in chunks]
        passages = [chunk["text"] for chunk in self.chunks]
        embeddings = embedder.embed_passages(passages)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype(np.float32))

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        if not self.chunks:
            return []
        k = min(top_k, len(self.chunks))
        query_embedding = self.embedder.embed_query(query).reshape(1, -1)
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        candidates: list[dict[str, Any]] = []
        for dense_rank, (score, index) in enumerate(zip(scores[0], indices[0]), start=1):
            if index < 0:
                continue
            chunk = self.chunks[int(index)]
            candidates.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "dense_rank": dense_rank,
                    "dense_score": float(score),
                    "global_index": chunk["global_index"],
                    "act": chunk["act"],
                    "scene": chunk["scene"],
                    "scene_title": chunk["scene_title"],
                    "retrieval_method": "dense_faiss",
                }
            )
        if self.reranker is None:
            for candidate in candidates:
                candidate["rank"] = candidate["dense_rank"]
                candidate["score"] = candidate["dense_score"]
            return candidates

        documents = [
            self.chunks[int(index)]["text"]
            for index in indices[0]
            if index >= 0
        ]
        rerank_scores = self.reranker.score(query, documents)
        for candidate, rerank_score in zip(candidates, rerank_scores):
            candidate["rerank_score"] = rerank_score
            candidate["score"] = rerank_score
            candidate["retrieval_method"] = "dense_faiss_reranked"

        reranked = sorted(
            candidates,
            key=lambda candidate: (
                -float(candidate["rerank_score"]),
                int(candidate["dense_rank"]),
            ),
        )
        for rank, candidate in enumerate(reranked, start=1):
            candidate["rank"] = rank
        return reranked


class BM25Retriever:
    """One-document Okapi BM25 retriever over the Hamlet chunks."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.chunks = [dict(chunk) for chunk in chunks]
        self.k1 = k1
        self.b = b
        self.term_frequencies = [
            Counter(tokenize_for_bm25(str(chunk["text"]))) for chunk in self.chunks
        ]
        self.doc_lengths = [sum(counter.values()) for counter in self.term_frequencies]
        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )
        document_frequencies: Counter[str] = Counter()
        for counter in self.term_frequencies:
            document_frequencies.update(counter.keys())
        document_count = len(self.chunks)
        self.idf = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    def _score_document(self, query_terms: Counter[str], index: int) -> float:
        if not self.avg_doc_length:
            return 0.0
        frequencies = self.term_frequencies[index]
        doc_length = self.doc_lengths[index]
        length_norm = self.k1 * (
            1 - self.b + self.b * doc_length / self.avg_doc_length
        )
        score = 0.0
        for term, query_frequency in query_terms.items():
            term_frequency = frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            numerator = term_frequency * (self.k1 + 1)
            denominator = term_frequency + length_norm
            score += query_frequency * self.idf.get(term, 0.0) * numerator / denominator
        return score

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        if not self.chunks:
            return []
        query_terms = Counter(tokenize_for_bm25(query))
        scored = [
            (index, self._score_document(query_terms, index))
            for index in range(len(self.chunks))
        ]
        ranked = sorted(
            scored,
            key=lambda item: (-item[1], int(self.chunks[item[0]]["global_index"])),
        )[: min(top_k, len(self.chunks))]

        results: list[dict[str, Any]] = []
        for rank, (index, score) in enumerate(ranked, start=1):
            chunk = self.chunks[index]
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "rank": rank,
                    "score": float(score),
                    "sparse_rank": rank,
                    "sparse_score": float(score),
                    "global_index": chunk["global_index"],
                    "act": chunk["act"],
                    "scene": chunk["scene"],
                    "scene_title": chunk["scene_title"],
                    "retrieval_method": "bm25",
                }
            )
        return results
