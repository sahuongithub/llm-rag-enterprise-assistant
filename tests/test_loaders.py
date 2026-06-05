import pytest

from rag_platform.ingestion.loaders import UnsupportedDocumentError, documents_from_bytes


def test_markdown_loader_creates_document() -> None:
    documents = documents_from_bytes("notes.md", b"# Architecture\nHybrid retrieval and reranking.")

    assert len(documents) == 1
    assert "Hybrid retrieval" in documents[0].text
    assert documents[0].metadata["filename"] == "notes.md"


def test_csv_loader_creates_document_per_row() -> None:
    documents = documents_from_bytes(
        "policies.csv",
        b"title,body\nLatency,p95 target under 900ms\nGrounding,Use citations\n",
    )

    assert len(documents) == 2
    assert "p95 target" in documents[0].text


def test_loader_rejects_unsupported_extension() -> None:
    with pytest.raises(UnsupportedDocumentError):
        documents_from_bytes("archive.zip", b"nope")
