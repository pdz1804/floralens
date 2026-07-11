"""Retrieval metrics: Recall@K, Precision@K, mAP@K, MRR (PRD §14.4).

All functions operate on a per-query "relevance list": an ordered list of
booleans (True = the ranked item at that position shares the query's class),
already sorted by descending similarity with the query's own id excluded.
This keeps the math independent of the vector store / dataset, so it is
directly unit-testable against hand-built fixtures with known answers.
"""
from __future__ import annotations


def recall_at_k(relevance: list[bool], k: int) -> float:
    """1.0 if at least one relevant item appears in the top-k, else 0.0."""
    if k <= 0:
        raise ValueError("k must be positive")
    return 1.0 if any(relevance[:k]) else 0.0


def precision_at_k(relevance: list[bool], k: int) -> float:
    """Fraction of the top-k that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive")
    top = relevance[:k]
    if not top:
        return 0.0
    return sum(top) / len(top)


def average_precision_at_k(relevance: list[bool], k: int) -> float:
    """Average precision over the top-k, rewarding correct ranking order.

    AP@K = (1 / num_relevant_in_top_k) * sum_i [rel_i * precision_at_i]
    Normalizes by the number of relevant items actually present in the top-k
    window (0 if none). Note this is NOT the textbook `min(K, total_relevant)`
    denominator: it cannot penalize relevant items ranked beyond K, because the
    harness already passes a top-k-truncated relevance list. Deliberate
    simplification, but be aware it is more lenient (a single hit at rank 1 with
    K=5 scores AP@5 = 1.0, not 0.2, when many relevant items exist).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    top = relevance[:k]
    num_relevant = sum(top)
    if num_relevant == 0:
        return 0.0
    running_hits = 0
    precision_sum = 0.0
    for i, rel in enumerate(top, start=1):
        if rel:
            running_hits += 1
            precision_sum += running_hits / i
    return precision_sum / num_relevant


def reciprocal_rank(relevance: list[bool]) -> float:
    """1 / rank of the first relevant item; 0.0 if none relevant."""
    for i, rel in enumerate(relevance, start=1):
        if rel:
            return 1.0 / i
    return 0.0


def mean_over_queries(per_query_values: list[float]) -> float:
    """Arithmetic mean of a per-query metric list; 0.0 for an empty list."""
    if not per_query_values:
        return 0.0
    return sum(per_query_values) / len(per_query_values)
