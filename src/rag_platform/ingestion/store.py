from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from rag_platform.core.models import Chunk


class JsonlChunkStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, chunks: Iterable[Chunk]) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with self.path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(
                    json.dumps(
                        {
                            "id": chunk.id,
                            "document_id": chunk.document_id,
                            "text": chunk.text,
                            "metadata": chunk.metadata,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
                count += 1
        return count

    def read(self) -> list[Chunk]:
        if not self.path.exists():
            return []
        chunks: list[Chunk] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                chunks.append(
                    Chunk(
                        id=payload["id"],
                        document_id=payload["document_id"],
                        text=payload["text"],
                        metadata=payload.get("metadata", {}),
                    )
                )
        return chunks

