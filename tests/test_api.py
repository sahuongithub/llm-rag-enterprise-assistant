from fastapi.testclient import TestClient

from rag_platform.api.app import create_app


def test_file_ingestion_and_query_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RAG_INDEX_PATH", str(tmp_path / "index.jsonl"))
    monkeypatch.setenv("RAG_UPLOAD_DIR", str(tmp_path / "uploads"))
    app = create_app()
    client = TestClient(app)

    ingest = client.post(
        "/ingest/files",
        files={"files": ("knowledge.txt", b"Hybrid retrieval reduces hallucinations.", "text/plain")},
    )
    assert ingest.status_code == 200
    assert ingest.json()["indexed_chunks"] == 1

    response = client.post("/query", json={"question": "What reduces hallucinations?"})
    assert response.status_code == 200
    assert "hallucinations" in response.json()["answer"].lower()


def test_api_key_is_enforced(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RAG_INDEX_PATH", str(tmp_path / "index.jsonl"))
    monkeypatch.setenv("RAG_API_KEY", "secret")
    app = create_app()
    client = TestClient(app)

    blocked = client.post("/query", json={"question": "What is indexed?"})
    allowed = client.post(
        "/query",
        json={"question": "What is indexed?"},
        headers={"x-api-key": "secret"},
    )

    assert blocked.status_code == 401
    assert allowed.status_code == 200
