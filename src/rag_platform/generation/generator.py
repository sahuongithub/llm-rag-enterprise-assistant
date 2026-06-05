from __future__ import annotations

import re

import httpx

from rag_platform.core.models import RetrievalResult


class LocalExtractiveGenerator:
    model_name = "local-extractive-grounded-generator"

    def generate(self, question: str, contexts: list[RetrievalResult]) -> str:
        if not contexts:
            return "I do not have enough retrieved context to answer that."

        query_terms = {term.lower() for term in re.findall(r"[a-zA-Z0-9]+", question)}
        selected: list[str] = []
        for result in contexts:
            sentences = re.split(r"(?<=[.!?])\s+", result.chunk.text)
            ranked = sorted(
                sentences,
                key=lambda sentence: len(query_terms.intersection(sentence.lower().split())),
                reverse=True,
            )
            if ranked and ranked[0]:
                selected.append(ranked[0].strip())
            if len(selected) >= 3:
                break
        body = " ".join(selected)
        return body or contexts[0].chunk.text[:600]


class VllmGenerator:
    def __init__(self, base_url: str, model: str, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self.timeout_s = timeout_s

    def generate(self, question: str, contexts: list[RetrievalResult]) -> str:
        context_block = "\n\n".join(
            f"[{index + 1}] {result.chunk.text}" for index, result in enumerate(contexts)
        )
        prompt = (
            "Answer using only the supplied context. Cite sources inline as [1], [2].\n\n"
            f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer:"
        )
        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.post(
                f"{self.base_url}/completions",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": 0.1,
                    "max_tokens": 512,
                },
            )
            response.raise_for_status()
            payload = response.json()
        return payload["choices"][0]["text"].strip()

