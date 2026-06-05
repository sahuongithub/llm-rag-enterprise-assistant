.PHONY: install test lint sample run docker

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests scripts

sample:
	python scripts/ingest_sample.py

run:
	uvicorn rag_platform.api.app:create_app --factory --reload

docker:
	docker compose up --build

