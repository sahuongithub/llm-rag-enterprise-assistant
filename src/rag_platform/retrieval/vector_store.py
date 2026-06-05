from __future__ import annotations

from rag_platform.core.models import Chunk
from rag_platform.retrieval.embeddings import HashingEmbedder, cosine


class InMemoryVectorStore:
    def __init__(self, embedder: HashingEmbedder | None = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []

    def add(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.vectors = [self.embedder.embed(chunk.text) for chunk in chunks]

    def search(self, query: str) -> list[float]:
        query_vector = self.embedder.embed(query)
        return [cosine(query_vector, vector) for vector in self.vectors]


class FaissVectorStore(InMemoryVectorStore):
    """FAISS-backed store when faiss-cpu/faiss-gpu is installed.

    The class falls back to the parent implementation when FAISS is unavailable,
    which keeps local tests simple and production configuration explicit.
    """

    def __init__(self, embedder: HashingEmbedder | None = None) -> None:
        super().__init__(embedder=embedder)
        try:
            import faiss  # type: ignore
        except ImportError:
            self.faiss = None
        else:
            self.faiss = faiss
            self.index = faiss.IndexFlatIP(self.embedder.dimensions)

    def add(self, chunks: list[Chunk]) -> None:
        super().add(chunks)
        if not self.faiss:
            return
        import numpy as np

        matrix = np.array(self.vectors, dtype="float32")
        self.index.reset()
        if len(matrix):
            self.index.add(matrix)

    def search(self, query: str) -> list[float]:
        if not self.faiss or not self.vectors:
            return super().search(query)
        import numpy as np

        query_vector = np.array([self.embedder.embed(query)], dtype="float32")
        scores, indices = self.index.search(query_vector, len(self.vectors))
        mapped = [0.0] * len(self.vectors)
        for score, index in zip(scores[0], indices[0]):
            if index >= 0:
                mapped[index] = float(score)
        return mapped

