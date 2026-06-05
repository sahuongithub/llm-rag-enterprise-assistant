from __future__ import annotations

from rag_platform.core.models import Chunk, RetrievalResult
from rag_platform.retrieval.sparse import BM25Index
from rag_platform.retrieval.vector_store import InMemoryVectorStore


def normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if high == low:
        return [0.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        vector_store: InMemoryVectorStore | None = None,
        dense_weight: float = 0.58,
    ) -> None:
        self.chunks = chunks
        self.vector_store = vector_store or InMemoryVectorStore()
        self.vector_store.add(chunks)
        self.sparse_index = BM25Index([chunk.text for chunk in chunks])
        self.dense_weight = dense_weight

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self.chunks:
            return []

        dense_scores = normalize(self.vector_store.search(query))
        sparse_scores = normalize(self.sparse_index.score(query))
        results: list[RetrievalResult] = []
        for index, chunk in enumerate(self.chunks):
            dense = dense_scores[index]
            sparse = sparse_scores[index]
            fused = dense * self.dense_weight + sparse * (1 - self.dense_weight)
            if query.lower() in chunk.text.lower():
                fused += 0.12
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=round(fused, 6),
                    dense_score=round(dense, 6),
                    sparse_score=round(sparse, 6),
                )
            )
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

