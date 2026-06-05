from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any

from rag_platform.core.models import Document, make_id


SUPPORTED_EXTENSIONS = {".csv", ".json", ".jsonl", ".md", ".txt"}


class UnsupportedDocumentError(ValueError):
    pass


def documents_from_bytes(filename: str, content: bytes) -> list[Document]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentError(
            f"Unsupported file type {suffix or '<none>'}. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    text = content.decode("utf-8", errors="replace")
    if suffix in {".txt", ".md"}:
        return [_document(filename, text, {"format": suffix.lstrip(".")})]
    if suffix == ".csv":
        return _from_csv(filename, text)
    if suffix == ".jsonl":
        return _from_jsonl(filename, text)
    return _from_json(filename, text)


def _document(filename: str, text: str, metadata: dict[str, Any] | None = None) -> Document:
    return Document(
        id=make_id("doc"),
        text=text,
        metadata={"filename": filename, **(metadata or {})},
    )


def _from_csv(filename: str, text: str) -> list[Document]:
    rows = csv.DictReader(StringIO(text))
    documents: list[Document] = []
    for index, row in enumerate(rows):
        row_text = "\n".join(f"{key}: {value}" for key, value in row.items() if value)
        if row_text.strip():
            documents.append(_document(filename, row_text, {"format": "csv", "row": index}))
    return documents


def _from_jsonl(filename: str, text: str) -> list[Document]:
    documents: list[Document] = []
    for index, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        payload = json.loads(line)
        documents.append(_from_json_payload(filename, payload, {"format": "jsonl", "row": index}))
    return documents


def _from_json(filename: str, text: str) -> list[Document]:
    payload = json.loads(text)
    if isinstance(payload, list):
        return [
            _from_json_payload(filename, item, {"format": "json", "row": index})
            for index, item in enumerate(payload)
        ]
    return [_from_json_payload(filename, payload, {"format": "json"})]


def _from_json_payload(filename: str, payload: Any, metadata: dict[str, Any]) -> Document:
    if isinstance(payload, dict):
        text = str(payload.get("text") or payload.get("content") or json.dumps(payload, indent=2))
        doc_id = str(payload.get("id") or make_id("doc"))
        extra = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        return Document(id=doc_id, text=text, metadata={"filename": filename, **metadata, **extra})
    return _document(filename, str(payload), metadata)

