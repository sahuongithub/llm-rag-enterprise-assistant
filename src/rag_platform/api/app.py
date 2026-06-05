from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_platform.core.config import Settings
from rag_platform.core.models import Document
from rag_platform.core.pipeline import AssistantPipeline
from rag_platform.generation.generator import LocalExtractiveGenerator, VllmGenerator
from rag_platform.ingestion.chunker import TextChunker
from rag_platform.ingestion.store import JsonlChunkStore
from rag_platform.retrieval.hybrid import HybridRetriever


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int | None = Field(default=None, ge=1, le=20)


class IngestRequest(BaseModel):
    documents: list[dict[str, object]]


def build_pipeline(settings: Settings) -> AssistantPipeline:
    chunks = JsonlChunkStore(settings.index_path).read()
    retriever = HybridRetriever(chunks, dense_weight=settings.dense_weight)
    generator = (
        VllmGenerator(settings.vllm_base_url, settings.vllm_model)
        if settings.generator_mode == "vllm"
        else LocalExtractiveGenerator()
    )
    return AssistantPipeline(retriever=retriever, generator=generator)


def create_app() -> FastAPI:
    settings = Settings.from_env()
    app = FastAPI(title="LLM-RAG Enterprise Assistant Platform", version="0.1.0")
    app.state.settings = settings
    app.state.pipeline = build_pipeline(settings)

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {"status": "ok", "chunks": len(app.state.pipeline.retriever.chunks)}

    @app.get("/metrics")
    def metrics() -> dict[str, object]:
        return {
            "rag_chunks_loaded": len(app.state.pipeline.retriever.chunks),
            "rag_generator_mode": settings.generator_mode,
        }

    @app.post("/query")
    def query(request: QueryRequest) -> dict[str, object]:
        answer = app.state.pipeline.answer(request.question, top_k=request.top_k or settings.top_k)
        return {
            "question": answer.question,
            "answer": answer.answer,
            "citations": answer.citations,
            "latency_ms": answer.latency_ms,
            "model": answer.model,
        }

    @app.post("/ingest")
    def ingest(request: IngestRequest) -> dict[str, object]:
        chunker = TextChunker(settings.chunk_size, settings.chunk_overlap)
        chunks = []
        for item in request.documents:
            document = Document(
                id=str(item["id"]),
                text=str(item["text"]),
                metadata=dict(item.get("metadata", {})),
            )
            chunks.extend(chunker.split(document))
        count = JsonlChunkStore(settings.index_path).write(chunks)
        app.state.pipeline = build_pipeline(settings)
        return {"indexed_chunks": count}

    return app

