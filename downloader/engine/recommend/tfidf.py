"""TF-IDF and cosine similarity, written by hand (spec A4).

A vector is a plain dict {word: weight}. Words that aren't in it have weight 0.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable


def fit_idf(docs: list[list[str]]) -> dict[str, float]:
    """How rare each word is across the documents: log(N / documents containing it)."""
    n = len(docs)
    df = Counter(word for doc in docs for word in set(doc))  # set(): a word counts once per document
    return {word: math.log(n / count) for word, count in df.items()}


def vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """tf × idf for one document, where tf = count / words in the document."""
    if not tokens:
        return {}
    counts = Counter(tokens)
    return {word: c / len(tokens) * idf[word] for word, c in counts.items() if word in idf}


def weighted_sum(items: Iterable[tuple[dict[str, float], float]]) -> dict[str, float]:
    """Σ weight × vector. A negative weight pulls words down (a dismissal)."""
    total: dict[str, float] = {}
    for vec, weight in items:
        for word, value in vec.items():
            total[word] = total.get(word, 0.0) + weight * value
    return total


def _length(v: dict[str, float]) -> float:
    return math.sqrt(sum(x * x for x in v.values()))


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """(a · b) / (|a| × |b|): 1 means the same direction, 0 means nothing in common."""
    la, lb = _length(a), _length(b)
    if la == 0 or lb == 0:
        return 0.0
    dot = sum(value * b.get(word, 0.0) for word, value in a.items())
    return dot / (la * lb)
