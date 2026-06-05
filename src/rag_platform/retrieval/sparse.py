from __future__ import annotations

import math
from collections import Counter

from rag_platform.retrieval.embeddings import tokenize


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents = [tokenize(text) for text in texts]
        self.lengths = [len(document) for document in self.documents]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.term_frequencies = [Counter(document) for document in self.documents]
        document_frequency: Counter[str] = Counter()
        for document in self.documents:
            document_frequency.update(set(document))
        total = max(len(self.documents), 1)
        self.idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def score(self, query: str) -> list[float]:
        query_terms = tokenize(query)
        scores: list[float] = []
        for index, frequencies in enumerate(self.term_frequencies):
            length = self.lengths[index] or 1
            score = 0.0
            for term in query_terms:
                tf = frequencies.get(term, 0)
                if tf == 0:
                    continue
                denominator = tf + self.k1 * (1 - self.b + self.b * length / self.average_length)
                score += self.idf.get(term, 0.0) * (tf * (self.k1 + 1)) / denominator
            scores.append(score)
        return scores

