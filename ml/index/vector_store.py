"""In-memory cosine-similarity vector store.

Holds id -> (vector, metadata) pairs. Vectors are expected to already be
L2-normalized (the embedding backbone guarantees this), so cosine similarity
reduces to a dot product. This is intentionally simple (YAGNI) — Phase 1 only
needs correctness at gallery scale (hundreds-thousands of vectors), not ANN.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class SearchResult:
    """One ranked nearest-neighbor hit: gallery specimen id, its cosine
    similarity score to the query, and its stored metadata (label etc)."""

    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class VectorIndex(Protocol):
    """Structural contract shared by every gallery vector-store backend.

    `VectorStore` (below) and `ml.index.pgvector_store.PgVectorStore` both
    satisfy this shape without inheriting from it — search_service depends on
    this Protocol, not on a concrete backend, so it can swap backends via
    config without either implementation knowing about the other.
    """

    model_version: str

    def add(self, id_: str, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        """Index one (id, vector, metadata) entry into the gallery."""
        ...

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 12,
        exclude_ids: set[str] | None = None,
    ) -> list[SearchResult]:
        """Return the top_k nearest neighbors to `vector` by cosine similarity,
        excluding any id in `exclude_ids`."""
        ...


class VectorStore:
    """A minimal in-memory vector index with cosine ranking."""

    def __init__(self, model_version: str) -> None:
        """Create an empty store tagged with `model_version` (carried through
        to search results so callers know which embedding space they came from)."""
        self.model_version = model_version
        self._ids: list[str] = []
        self._vectors: list[np.ndarray] = []
        self._metadata: dict[str, dict[str, Any]] = {}
        self._matrix: np.ndarray | None = None  # cached stacked matrix

    def add(self, id_: str, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        """Index one (id, vector, metadata) entry.

        The vector is L2-normalized before storage (defensive, in case it
        arrives not already unit-norm) so `query`'s dot product is a true
        cosine similarity. Raises on a duplicate id or a zero vector.
        """
        if id_ in self._metadata:
            raise ValueError(f"duplicate id in vector store: {id_}")
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError(f"zero vector for id {id_} cannot be normalized")
        # Defensive re-normalization in case caller passed a near-unit vector.
        vector = vector / norm
        self._ids.append(id_)
        self._vectors.append(vector)
        self._metadata[id_] = metadata or {}
        self._matrix = None  # invalidate cache

    def __len__(self) -> int:
        """Number of vectors currently indexed."""
        return len(self._ids)

    def _stacked(self) -> np.ndarray:
        if self._matrix is None:
            if not self._vectors:
                self._matrix = np.zeros((0, 0), dtype=np.float32)
            else:
                self._matrix = np.stack(self._vectors, axis=0)
        return self._matrix

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 12,
        exclude_ids: set[str] | None = None,
    ) -> list[SearchResult]:
        """Return the top_k nearest neighbors by cosine similarity.

        exclude_ids lets a query exclude its own embedding from its own
        results, per the retrieval eval protocol (PRD §14.4).
        """
        if len(self._ids) == 0:
            return []
        query_vec = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm == 0:
            raise ValueError("query vector must be non-zero")
        query_vec = query_vec / norm

        matrix = self._stacked()
        scores = matrix @ query_vec  # cosine similarity (both sides unit-norm)
        # float32 rounding can push a near-identical vector's dot product
        # a hair outside [-1, 1]; clamp so downstream consumers (e.g. an API
        # schema asserting score <= 1.0) never see e.g. 1.0000001.
        scores = np.clip(scores, -1.0, 1.0)

        exclude_ids = exclude_ids or set()
        candidates = [
            (self._ids[i], float(scores[i]))
            for i in range(len(self._ids))
            if self._ids[i] not in exclude_ids
        ]
        # Sort by score descending, breaking ties by id ascending so the order
        # is deterministic and matches the pgvector backend's `ORDER BY
        # <distance>, id` — both backends return the same top_k on ties.
        candidates.sort(key=lambda pair: (-pair[1], pair[0]))
        top = candidates[:top_k]
        return [
            SearchResult(id=id_, score=score, metadata=self._metadata[id_])
            for id_, score in top
        ]
