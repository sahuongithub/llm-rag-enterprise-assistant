FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAG_INDEX_PATH=/app/data/index.jsonl

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
RUN pip install --no-cache-dir .
RUN python scripts/ingest_sample.py

EXPOSE 8000
CMD ["uvicorn", "rag_platform.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

