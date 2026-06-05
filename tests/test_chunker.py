from rag_platform.core.models import Document
from rag_platform.ingestion.chunker import TextChunker


def test_chunker_splits_with_overlap_metadata() -> None:
    document = Document(id="doc", text=" ".join(["token"] * 80), metadata={"source": "unit"})
    chunks = TextChunker(chunk_size=80, overlap=10).split(document)

    assert len(chunks) > 1
    assert chunks[0].document_id == "doc"
    assert chunks[0].metadata["source"] == "unit"
    assert chunks[1].metadata["chunk_index"] == 1

