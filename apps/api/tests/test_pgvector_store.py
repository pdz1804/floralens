"""Tests for the opt-in pgvector gallery backend.

Two tiers:
  - Pure-unit / no-DB tests (always run): the `VectorIndex` Protocol contract
    and that `get_gallery_store()` stays on the in-memory `VectorStore` by
    default (settings.vector_store == "in_memory", unset).
  - Live-DB integration tests against `PgVectorStore` itself, gated on
    FLORALENS_TEST_PGVECTOR_DSN — skipped whenever that env var is unset
    (e.g. in this sandboxed dev environment, which cannot reach a DB).
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np
import pytest

from ml.index.vector_store import VectorIndex, VectorStore

EMBEDDINGS_CACHE = Path("ml/data/embeddings_cache/metadata.json")
PGVECTOR_DSN = os.environ.get("FLORALENS_TEST_PGVECTOR_DSN")


# --------------------------------------------------------------------------
# Pure-unit tests (no DB, no embeddings cache required) — always run.
# --------------------------------------------------------------------------


def test_vector_store_satisfies_vector_index_protocol():
    store = VectorStore(model_version="baseline")
    assert isinstance(store, VectorIndex)


@pytest.mark.skipif(not EMBEDDINGS_CACHE.exists(), reason="embeddings cache not built yet")
def test_get_gallery_store_defaults_to_in_memory():
    from apps.api.app import search_service
    from apps.api.app.config import settings

    assert settings.vector_store == "in_memory"
    search_service.reset_gallery_store_cache()
    try:
        store = search_service.get_gallery_store()
        assert isinstance(store, VectorStore)
    finally:
        search_service.reset_gallery_store_cache()


# --------------------------------------------------------------------------
# Live-DB integration tests — require FLORALENS_TEST_PGVECTOR_DSN.
# --------------------------------------------------------------------------


@pytest.mark.skipif(not PGVECTOR_DSN, reason="no pgvector DSN")
class TestPgVectorStoreIntegration:
    def _make_store(self):
        from ml.index.pgvector_store import PgVectorStore

        table = f"fl_gallery_test_{uuid.uuid4().hex}"
        store = PgVectorStore("test", PGVECTOR_DSN, dim=4, table=table)
        return store, table

    def _drop(self, store, table) -> None:
        try:
            store._conn.execute(f"DROP TABLE IF EXISTS {table};")
        finally:
            store.close()

    def test_add_and_query_orders_by_cosine_similarity(self):
        store, table = self._make_store()
        try:
            store.add("a", np.array([1.0, 0.0, 0.0, 0.0]), {"label_name": "a"})
            store.add("b", np.array([0.9, 0.1, 0.0, 0.0]), {"label_name": "b"})
            store.add("c", np.array([0.0, 1.0, 0.0, 0.0]), {"label_name": "c"})

            results = store.query(np.array([1.0, 0.0, 0.0, 0.0]), top_k=3)

            assert [r.id for r in results] == ["a", "b", "c"]
            for r in results:
                assert -1.0 <= r.score <= 1.0
            assert results[0].score == pytest.approx(1.0, abs=1e-4)
            assert results[0].metadata == {"label_name": "a"}
        finally:
            self._drop(store, table)

    def test_exclude_ids_removes_row(self):
        store, table = self._make_store()
        try:
            store.add("a", np.array([1.0, 0.0, 0.0, 0.0]))
            store.add("b", np.array([0.9, 0.1, 0.0, 0.0]))

            results = store.query(
                np.array([1.0, 0.0, 0.0, 0.0]), top_k=5, exclude_ids={"a"}
            )

            assert [r.id for r in results] == ["b"]
        finally:
            self._drop(store, table)

    def test_upsert_replaces_vector(self):
        store, table = self._make_store()
        try:
            store.add("a", np.array([1.0, 0.0, 0.0, 0.0]), {"v": 1})
            assert store.count() == 1

            # Re-add the same id -> ON CONFLICT DO UPDATE, not a second row.
            store.add("a", np.array([0.0, 1.0, 0.0, 0.0]), {"v": 2})
            assert store.count() == 1

            results = store.query(np.array([0.0, 1.0, 0.0, 0.0]), top_k=1)
            assert results[0].id == "a"
            assert results[0].score == pytest.approx(1.0, abs=1e-4)
            assert results[0].metadata == {"v": 2}
        finally:
            self._drop(store, table)

    def test_query_scoped_to_model_version(self):
        from ml.index.pgvector_store import PgVectorStore

        table = f"fl_gallery_test_{uuid.uuid4().hex}"
        store_a = PgVectorStore("test-a", PGVECTOR_DSN, dim=4, table=table)
        store_b = PgVectorStore("test-b", PGVECTOR_DSN, dim=4, table=table)
        try:
            store_a.add("a", np.array([1.0, 0.0, 0.0, 0.0]))
            store_b.add("b", np.array([1.0, 0.0, 0.0, 0.0]))

            results_a = store_a.query(np.array([1.0, 0.0, 0.0, 0.0]), top_k=5)
            assert [r.id for r in results_a] == ["a"]
            assert store_a.count() == 1
        finally:
            self._drop(store_a, table)
            store_b.close()

    def test_ingest_replaces_atomically_and_matches_count(self):
        store, table = self._make_store()
        try:
            store.add("stale", np.array([0.0, 0.0, 0.0, 1.0]))  # pre-existing row
            records = [
                ("a", np.array([1.0, 0.0, 0.0, 0.0]), {"label_name": "a"}),
                ("b", np.array([0.0, 1.0, 0.0, 0.0]), {"label_name": "b"}),
            ]
            store.ingest(records)
            # ingest is a full replace for this model_version: exactly `records`.
            assert store.count() == len(records)
            ids = {r.id for r in store.query(np.array([1.0, 0.0, 0.0, 0.0]), top_k=10)}
            assert ids == {"a", "b"}  # 'stale' gone
        finally:
            self._drop(store, table)

    def test_ties_are_broken_by_id(self):
        store, table = self._make_store()
        try:
            # Identical vectors => identical cosine distance; ordering must fall
            # back to id ascending, deterministically (parity with in-memory).
            for vid in ("c", "a", "b"):
                store.add(vid, np.array([1.0, 0.0, 0.0, 0.0]))
            results = store.query(np.array([1.0, 0.0, 0.0, 0.0]), top_k=3)
            assert [r.id for r in results] == ["a", "b", "c"]
        finally:
            self._drop(store, table)

    def test_non_positive_top_k_returns_empty(self):
        store, table = self._make_store()
        try:
            store.add("a", np.array([1.0, 0.0, 0.0, 0.0]))
            assert store.query(np.array([1.0, 0.0, 0.0, 0.0]), top_k=0) == []
            assert store.query(np.array([1.0, 0.0, 0.0, 0.0]), top_k=-1) == []
        finally:
            self._drop(store, table)
