"""Retrieval evaluation harness (PRD §14.4).

Wires together: a vector store already indexed with a gallery partition, a
query set (val or test specimens with precomputed embeddings), and the pure
metric functions in `ml.eval.metrics`. Produces an EvalReport dict suitable
for JSON serialization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import silhouette_score

from ml.eval.metrics import (
    average_precision_at_k,
    mean_over_queries,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ml.index.vector_store import VectorStore

K_VALUES = (1, 5, 10)


@dataclass(frozen=True)
class EmbeddedSpecimen:
    id: str
    label: int
    label_name: str
    vector: np.ndarray


def run_retrieval_eval(
    queries: list[EmbeddedSpecimen],
    gallery_store: VectorStore,
    k_values: tuple[int, ...] = K_VALUES,
) -> dict[str, Any]:
    """Evaluate a query set against an already-indexed gallery VectorStore.

    Each query's own id is excluded from its own results (it should not be
    in the gallery at all by construction, but exclusion is defensive per
    §14.4). Relevance = same class label as the query.
    """
    max_k = max(k_values)
    per_query_metrics: list[dict[str, float]] = []

    for q in queries:
        results = gallery_store.query(q.vector, top_k=max_k, exclude_ids={q.id})
        relevance = [r.metadata["label"] == q.label for r in results]
        # Pad relevance to max_k with False if the gallery had fewer than max_k items.
        if len(relevance) < max_k:
            relevance = relevance + [False] * (max_k - len(relevance))

        row = {"query_id": q.id, "mrr": reciprocal_rank(relevance)}
        for k in k_values:
            row[f"recall@{k}"] = recall_at_k(relevance, k)
            row[f"precision@{k}"] = precision_at_k(relevance, k)
            row[f"map@{k}"] = average_precision_at_k(relevance, k)
        per_query_metrics.append(row)

    aggregate: dict[str, float] = {"mrr": mean_over_queries([r["mrr"] for r in per_query_metrics])}
    for k in k_values:
        aggregate[f"recall@{k}"] = mean_over_queries([r[f"recall@{k}"] for r in per_query_metrics])
        aggregate[f"precision@{k}"] = mean_over_queries([r[f"precision@{k}"] for r in per_query_metrics])
        aggregate[f"map@{k}"] = mean_over_queries([r[f"map@{k}"] for r in per_query_metrics])

    aggregate["silhouette"] = _safe_silhouette(queries)
    aggregate["num_queries"] = len(queries)
    aggregate["num_gallery"] = len(gallery_store)

    return {"aggregate": aggregate, "per_query": per_query_metrics}


def _safe_silhouette(items: list[EmbeddedSpecimen]) -> float | None:
    """Mean silhouette score over classes; None if degenerate (<2 classes)."""
    labels = [item.label for item in items]
    if len(set(labels)) < 2 or len(items) < 3:
        return None
    matrix = np.stack([item.vector for item in items], axis=0)
    try:
        return float(silhouette_score(matrix, labels, metric="cosine"))
    except ValueError:
        return None
