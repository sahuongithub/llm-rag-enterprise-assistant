from __future__ import annotations

import json
from statistics import mean

from rag_platform.api.app import build_pipeline
from rag_platform.core.config import Settings
from rag_platform.evaluation.evaluate import EvaluationCase


CASES = [
    EvaluationCase(
        question="How does the platform reduce hallucinations?",
        expected_terms={"hybrid", "reranking", "grounding"},
    ),
    EvaluationCase(
        question="How does vLLM help serving cost?",
        expected_terms={"paged", "quantization", "gpu"},
    ),
    EvaluationCase(
        question="How is the API operated in Kubernetes?",
        expected_terms={"health", "metrics", "autoscaling"},
    ),
]


def main() -> None:
    settings = Settings.from_env()
    pipeline = build_pipeline(settings)
    answers = [pipeline.answer(case.question) for case in CASES]
    hits = sum(
        any(term.lower() in answer.answer.lower() for term in case.expected_terms)
        for case, answer in zip(CASES, answers)
    )
    latencies = sorted(answer.latency_ms for answer in answers)
    report = {
        "hit_rate": round(hits / len(CASES), 4),
        "average_latency_ms": round(mean(latencies), 2),
    }
    report["p95_latency_ms"] = sorted(answer.latency_ms for answer in answers)[-1]
    report["average_citations"] = round(mean(len(answer.citations) for answer in answers), 2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
