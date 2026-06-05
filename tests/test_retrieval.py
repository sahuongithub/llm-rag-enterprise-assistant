from rag_platform.core.models import Chunk
from rag_platform.retrieval.hybrid import HybridRetriever


def test_hybrid_retrieval_prefers_exact_and_semantic_matches() -> None:
    chunks = [
        Chunk(id="1", document_id="a", text="Kubernetes autoscaling protects API latency."),
        Chunk(id="2", document_id="b", text="vLLM paged attention improves GPU inference cost."),
        Chunk(id="3", document_id="c", text="Policy documents require source citations."),
    ]
    retriever = HybridRetriever(chunks)

    results = retriever.search("How does vLLM lower GPU serving cost?", top_k=2)

    assert results[0].chunk.id == "2"
    assert results[0].score >= results[1].score

