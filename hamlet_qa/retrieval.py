"""Dense FAISS retrieval for the Hamlet QA harness."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from hamlet_qa.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_TOP_K


class EmbedderLike(Protocol):
    def embed_passages(self, passages: list[str]) -> np.ndarray:
        ...

    def embed_query(self, query: str) -> np.ndarray:
        ...


class SnowflakeEmbedder:
    """SentenceTransformers wrapper with Snowflake's query prompt."""

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
        self.uses_prompt_name_query = "snowflake-arctic-embed" in model_name.lower()

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
        if self.uses_prompt_name_query:
            kwargs["prompt_name"] = "query"
        return self.model.encode(query, **kwargs).astype(np.float32)


class DenseRetriever:
    """One-document FAISS inner-product retriever over normalized embeddings."""

    def __init__(self, embedder: EmbedderLike, chunks: list[dict[str, Any]]):
        import faiss

        self.embedder = embedder
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
        trace: list[dict[str, Any]] = []
        for rank, (score, index) in enumerate(zip(scores[0], indices[0]), start=1):
            if index < 0:
                continue
            chunk = self.chunks[int(index)]
            trace.append(
                {
                    "rank": rank,
                    "chunk_id": chunk["chunk_id"],
                    "score": float(score),
                    "global_index": chunk["global_index"],
                    "act": chunk["act"],
                    "scene": chunk["scene"],
                    "scene_title": chunk["scene_title"],
                }
            )
        return trace
