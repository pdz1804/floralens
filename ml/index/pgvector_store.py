"""Opt-in pgvector-backed gallery vector store.

Mirrors `ml.index.vector_store.VectorStore`'s surface (same `SearchResult`,
same `add`/`query`/`model_version` contract — see `VectorIndex` in that
module) but persists vectors in Postgres via the `pgvector` extension instead
of an in-process numpy matrix. This lets a gallery survive process restarts
and be queried by more than one API worker.

Synchronous by design: `search_service.search_image` calls `store.query(...)`
on the request-handling thread exactly like the in-memory store today, so
this uses psycopg 3's sync API (no asyncio plumbing needed to fit the
existing call site).

Dependencies (`psycopg[binary]`, `pgvector`) are optional — nothing in this
module is imported unless `FLORALENS_VECTOR_STORE=pgvector` is set (see
`apps.api.app.search_service.get_gallery_store`), and even a bare
`import ml.index.pgvector_store` must not fail when those packages are
absent. All third-party imports are therefore deferred into `__init__`.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ml.index.vector_store import SearchResult

# Identifiers are validated (not user input at call time, but defensive
# against a misconfigured table name reaching raw SQL) before being
# interpolated into DDL/DML — psycopg placeholders can't parametrize
# identifiers, only values.
_VALID_IDENT = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)


def _validate_identifier(name: str, kind: str) -> None:
    if not name or name[0].isdigit() or any(ch not in _VALID_IDENT for ch in name):
        raise ValueError(f"invalid {kind} identifier: {name!r}")


class PgVectorStore:
    """Postgres/pgvector-backed gallery index satisfying `VectorIndex`.

    Rows are tagged with `model_version` so several model versions of the SAME
    embedding dimension can coexist in one table, each queried in isolation.
    The `embedding` column has a fixed dimension (`dim`, set on first create),
    so model versions whose vectors differ in length need separate tables — a
    mismatched insert fails loudly rather than corrupting data.
    """

    def __init__(
        self,
        model_version: str,
        dsn: str,
        *,
        dim: int = 1024,
        table: str = "fl_gallery",
    ) -> None:
        _validate_identifier(table, "table")
        import psycopg  # local import: keep this module importable without psycopg installed
        from pgvector.psycopg import register_vector

        self.model_version = model_version
        self._dim = dim
        self._table = table
        self._dsn = dsn
        self._psycopg = psycopg  # kept for the query-time reconnect path

        self._conn = self._connect()
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id text NOT NULL,
                model_version text NOT NULL,
                embedding vector({dim}) NOT NULL,
                metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                PRIMARY KEY (id, model_version)
            );
            """
        )

    def _connect(self):
        """Open an autocommit connection with the pgvector codec + extension."""
        from pgvector.psycopg import register_vector

        conn = self._psycopg.connect(self._dsn, autocommit=True)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        register_vector(conn)
        return conn

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("zero vector cannot be normalized")
        return vector / norm

    def add(self, id_: str, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        import json

        vector = self._normalize(vector)
        self._conn.execute(
            f"""
            INSERT INTO {self._table} (id, model_version, embedding, metadata)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id, model_version) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata;
            """,
            (id_, self.model_version, vector, json.dumps(metadata or {})),
        )

    def ingest(self, records: list[tuple[str, np.ndarray, dict[str, Any]]]) -> None:
        """Atomically replace this model_version's rows with `records`.

        Runs as ONE transaction: the delete + every insert commit together or
        not at all. A failure part-way (dropped connection, bad row) rolls the
        whole thing back, so the table is never left partially populated — which
        would otherwise look "already ingested" to the count() gate and serve an
        incomplete gallery forever.
        """
        with self._conn.transaction():
            self._conn.execute(
                f"DELETE FROM {self._table} WHERE model_version = %s;", (self.model_version,)
            )
            for id_, vector, meta in records:
                self.add(id_, vector, meta)

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 12,
        exclude_ids: set[str] | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []
        try:
            return self._query(vector, top_k, exclude_ids)
        except self._psycopg.OperationalError:
            # The single long-lived connection dropped (DB restart / idle
            # timeout / transient network blip). Reconnect once and retry so a
            # blip doesn't brick /api/search for the life of the process.
            self._conn = self._connect()
            return self._query(vector, top_k, exclude_ids)

    def _query(
        self,
        vector: np.ndarray,
        top_k: int,
        exclude_ids: set[str] | None,
    ) -> list[SearchResult]:
        query_vec = self._normalize(vector)

        # Secondary `, id` sort makes ties deterministic and identical to the
        # in-memory backend's stable-sort order, so both return the same top_k.
        if exclude_ids:
            sql = f"""
                SELECT id, metadata, 1 - (embedding <=> %s) AS score
                FROM {self._table}
                WHERE model_version = %s AND id <> ALL(%s)
                ORDER BY embedding <=> %s, id
                LIMIT %s;
            """
            params = (query_vec, self.model_version, list(exclude_ids), query_vec, top_k)
        else:
            sql = f"""
                SELECT id, metadata, 1 - (embedding <=> %s) AS score
                FROM {self._table}
                WHERE model_version = %s
                ORDER BY embedding <=> %s, id
                LIMIT %s;
            """
            params = (query_vec, self.model_version, query_vec, top_k)

        rows = self._conn.execute(sql, params).fetchall()
        results = []
        for id_, metadata, score in rows:
            score = float(np.clip(score, -1.0, 1.0))
            results.append(SearchResult(id=id_, score=score, metadata=metadata or {}))
        return results

    def count(self) -> int:
        """Rows currently indexed for `self.model_version` (used by
        search_service to decide whether the gallery needs (re)ingestion)."""
        row = self._conn.execute(
            f"SELECT count(*) FROM {self._table} WHERE model_version = %s;",
            (self.model_version,),
        ).fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._conn.close()
