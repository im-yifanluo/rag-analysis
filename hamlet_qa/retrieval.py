"""Dense FAISS retrieval and reranking for the Hamlet QA harness."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from hamlet_qa.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_TOP_K,
)


class EmbedderLike(Protocol):
    def embed_passages(self, passages: list[str]) -> np.ndarray:
        ...

    def embed_query(self, query: str) -> np.ndarray:
        ...


class RerankerLike(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]:
        ...


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
