from __future__ import annotations

from pathlib import Path

from rag_platform.core.config import Settings
from rag_platform.core.models import Document
from rag_platform.ingestion.chunker import TextChunker
from rag_platform.ingestion.store import JsonlChunkStore


SAMPLE_DOCUMENTS = [
    Document(
        id="architecture",
        text=(
            "The enterprise RAG platform uses hybrid dense and sparse retrieval with reranking "
            "to improve grounding and reduce hallucinations. Dense retrieval captures semantic "
            "similarity while BM25-style sparse retrieval preserves exact product, policy, and "
            "identifier matches."
        ),
        metadata={"source": "sample", "owner": "platform"},
    ),
    Document(
        id="serving",
        text=(
            "The generation layer can call a vLLM OpenAI-compatible endpoint. vLLM paged "
            "attention improves GPU memory utilization for high-throughput inference, and INT8 "
            "quantization can reduce serving cost when quality gates are met."
        ),
        metadata={"source": "sample", "owner": "ml-infra"},
    ),
    Document(
        id="operations",
        text=(
            "Kubernetes deployment manifests define API replicas, resource requests, service "
            "routing, and autoscaling. The API exposes health and metrics endpoints so production "
            "checks can detect empty indexes or degraded generation backends."
        ),
        metadata={"source": "sample", "owner": "sre"},
    ),
]


def main() -> None:
    settings = Settings.from_env()
    chunker = TextChunker(settings.chunk_size, settings.chunk_overlap)
    chunks = [chunk for document in SAMPLE_DOCUMENTS for chunk in chunker.split(document)]
    count = JsonlChunkStore(settings.index_path).write(chunks)
    print(f"Wrote {count} chunks to {Path(settings.index_path).resolve()}")


if __name__ == "__main__":
    main()

