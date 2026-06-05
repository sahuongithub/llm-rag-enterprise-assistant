from __future__ import annotations

from rag_platform.core.models import Answer, Timer
from rag_platform.generation.generator import LocalExtractiveGenerator, VllmGenerator
from rag_platform.retrieval.hybrid import HybridRetriever


class AssistantPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        generator: LocalExtractiveGenerator | VllmGenerator | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator or LocalExtractiveGenerator()

    def answer(self, question: str, top_k: int = 5) -> Answer:
        with Timer() as timer:
            contexts = self.retriever.search(question, top_k=top_k)
            response = self.generator.generate(question, contexts)
        citations = [
            {
                "chunk_id": result.chunk.id,
                "document_id": result.chunk.document_id,
                "score": result.score,
                "metadata": result.chunk.metadata,
            }
            for result in contexts
        ]
        return Answer(
            question=question,
            answer=response,
            citations=citations,
            latency_ms=round(timer.elapsed_ms, 2),
            model=self.generator.model_name,
        )

