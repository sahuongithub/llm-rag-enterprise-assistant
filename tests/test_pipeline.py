from rag_platform.core.models import Chunk
from rag_platform.core.pipeline import AssistantPipeline
from rag_platform.retrieval.hybrid import HybridRetriever


def test_pipeline_returns_grounded_answer_and_citations() -> None:
    chunks = [
        Chunk(
            id="chunk-1",
            document_id="doc-1",
            text="Hybrid retrieval reduces hallucinations by combining dense search, BM25, and reranking.",
        )
    ]
    pipeline = AssistantPipeline(HybridRetriever(chunks))

    answer = pipeline.answer("How are hallucinations reduced?")

    assert "hallucinations" in answer.answer.lower()
    assert answer.citations[0]["chunk_id"] == "chunk-1"
    assert answer.latency_ms >= 0

