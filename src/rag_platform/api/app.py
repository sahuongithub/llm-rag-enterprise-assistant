from __future__ import annotations

import json
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_platform.core.config import Settings
from rag_platform.core.models import Document
from rag_platform.core.pipeline import AssistantPipeline
from rag_platform.generation.generator import LocalExtractiveGenerator, VllmGenerator
from rag_platform.ingestion.chunker import TextChunker
from rag_platform.ingestion.loaders import UnsupportedDocumentError, documents_from_bytes
from rag_platform.ingestion.store import JsonlChunkStore
from rag_platform.observability.metrics import CounterRegistry
from rag_platform.retrieval.hybrid import HybridRetriever


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int | None = Field(default=None, ge=1, le=20)


class IngestRequest(BaseModel):
    documents: list[dict[str, object]]


def require_api_key(settings: Settings):
    def dependency(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    return dependency


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
    app.state.metrics = CounterRegistry()
    auth = Depends(require_api_key(settings))
    static_dir = Path(__file__).resolve().parents[1] / "web"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {"status": "ok", "chunks": len(app.state.pipeline.retriever.chunks)}

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        body = app.state.metrics.prometheus(
            {
                "rag_chunks_loaded": len(app.state.pipeline.retriever.chunks),
                "rag_generator_mode": settings.generator_mode,
            }
        )
        return PlainTextResponse(body, media_type="text/plain; version=0.0.4")

    @app.post("/query")
    def query(request: QueryRequest, _: None = auth) -> dict[str, object]:
        answer = app.state.pipeline.answer(request.question, top_k=request.top_k or settings.top_k)
        app.state.metrics.increment("rag_queries_total")
        return {
            "question": answer.question,
            "answer": answer.answer,
            "citations": answer.citations,
            "latency_ms": answer.latency_ms,
            "model": answer.model,
        }

    @app.post("/ingest")
    def ingest(request: IngestRequest, _: None = auth) -> dict[str, object]:
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
        app.state.metrics.increment("rag_ingest_jobs_total")
        return {"indexed_chunks": count}

    @app.post("/ingest/files")
    async def ingest_files(files: list[UploadFile] = File(...), _: None = auth) -> dict[str, object]:
        chunker = TextChunker(settings.chunk_size, settings.chunk_overlap)
        documents: list[Document] = []
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        for upload in files:
            content = await upload.read()
            if not upload.filename:
                continue
            filename = Path(upload.filename).name
            try:
                loaded = documents_from_bytes(filename, content)
            except UnsupportedDocumentError as error:
                raise HTTPException(status_code=415, detail=str(error)) from error
            except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
                raise HTTPException(status_code=400, detail=f"Could not parse {filename}") from error
            (settings.upload_dir / filename).write_bytes(content)
            documents.extend(loaded)
        chunks = [chunk for document in documents for chunk in chunker.split(document)]
        count = JsonlChunkStore(settings.index_path).write(chunks)
        app.state.pipeline = build_pipeline(settings)
        app.state.metrics.increment("rag_ingest_jobs_total")
        return {"indexed_documents": len(documents), "indexed_chunks": count}

    return app
