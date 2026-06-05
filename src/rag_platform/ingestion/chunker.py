from __future__ import annotations

from rag_platform.core.models import Chunk, Document, make_id


class TextChunker:
    def __init__(self, chunk_size: int = 900, overlap: int = 120) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: Document) -> list[Chunk]:
        text = " ".join(document.text.split())
        if not text:
            return []

        chunks: list[Chunk] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start + self.chunk_size * 0.6:
                    end = boundary
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        id=make_id("chk"),
                        document_id=document.id,
                        text=chunk_text,
                        metadata={**document.metadata, "chunk_index": len(chunks)},
                    )
                )
            if end == len(text):
                break
            start = max(end - self.overlap, 0)
        return chunks

