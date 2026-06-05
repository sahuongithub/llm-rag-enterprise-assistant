from __future__ import annotations

from dataclasses import dataclass

from rag_platform.core.pipeline import AssistantPipeline


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    expected_terms: set[str]


def evaluate_grounding(pipeline: AssistantPipeline, cases: list[EvaluationCase]) -> dict[str, float]:
    if not cases:
        return {"hit_rate": 0.0, "average_latency_ms": 0.0}

    hits = 0
    total_latency = 0.0
    for case in cases:
        answer = pipeline.answer(case.question)
        total_latency += answer.latency_ms
        answer_text = answer.answer.lower()
        if any(term.lower() in answer_text for term in case.expected_terms):
            hits += 1

    return {
        "hit_rate": round(hits / len(cases), 4),
        "average_latency_ms": round(total_latency / len(cases), 2),
    }

