from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from rag_platform.core.models import Chunk


class JsonlChunkStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, chunks: Iterable[Chunk]) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = self.path.with_suffix(".manifest.json")
        count = 0
        checksum = hashlib.sha256()
        with self.path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                line = json.dumps(
                    {
                        "id": chunk.id,
                        "document_id": chunk.document_id,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                    },
                    ensure_ascii=True,
                )
                checksum.update(line.encode("utf-8"))
                handle.write(line + "\n")
                count += 1
        manifest_path.write_text(
            json.dumps({"chunks": count, "sha256": checksum.hexdigest()}, indent=2),
            encoding="utf-8",
        )
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
