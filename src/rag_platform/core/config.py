from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    index_path: Path = Path("./data/index.jsonl")
    upload_dir: Path = Path("./data/uploads")
    api_key: str | None = None
    generator_mode: str = "local"
    vllm_base_url: str = "http://vllm:8000/v1"
    vllm_model: str = "enterprise-llm-7b-lora"
    top_k: int = 5
    dense_weight: float = 0.58
    chunk_size: int = 900
    chunk_overlap: int = 120

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            index_path=Path(os.getenv("RAG_INDEX_PATH", "./data/index.jsonl")),
            upload_dir=Path(os.getenv("RAG_UPLOAD_DIR", "./data/uploads")),
            api_key=os.getenv("RAG_API_KEY"),
            generator_mode=os.getenv("RAG_GENERATOR_MODE", "local"),
            vllm_base_url=os.getenv("VLLM_BASE_URL", "http://vllm:8000/v1"),
            vllm_model=os.getenv("VLLM_MODEL", "enterprise-llm-7b-lora"),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
            dense_weight=float(os.getenv("RAG_DENSE_WEIGHT", "0.58")),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "900")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "120")),
        )
