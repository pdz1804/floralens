"""Garden service: persists the user's saved specimens ("My Garden", PRD
Phase 7 / `GardenPlant`) so the collection survives an API restart.

**Storage choice:** SQLite (stdlib `sqlite3`, no new dependency) in a single
file under `GARDEN_DB_PATH` (default `apps/api/data/garden.db`). A flat JSON
file would also work for this small, single-table shape, but SQLite gives
concurrency-safe writes (FastAPI's threadpool can run request handlers on
multiple threads) and an upsert-on-save via `ON CONFLICT`, both of which a
hand-rolled JSON read/modify/write would need to reimplement. The DB file is
created lazily on first use and is gitignored (runtime state, not source).

One row per `(user_id, specimen_id)` — saving an already-saved specimen just
refreshes `saved_at` rather than erroring or duplicating. `user_id` isolates
each caller's garden (per-user auth scaffold, see auth.py); every function
here defaults it to `DEFAULT_USER` ("public") so callers that never resolved
a real identity (auth off) keep today's single-user behavior unchanged.
"""
from __future__ import annotations

import functools
import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from apps.api.app.auth import DEFAULT_USER
from apps.api.app.config import settings

# Mutable at module scope (not a frozen default-arg) so tests can repoint it
# to a tmp_path via `monkeypatch.setattr(garden_service, "_DB_PATH", ...)`,
# matching this codebase's existing `monkeypatch.setattr(search_service, ...)`
# test convention rather than adding a bespoke setter API.
_DB_PATH = Path(os.environ.get("GARDEN_DB_PATH", "apps/api/data/garden.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS garden (
    user_id TEXT NOT NULL DEFAULT 'public',
    specimen_id TEXT NOT NULL,
    label_name TEXT NOT NULL,
    saved_at TEXT NOT NULL,
    PRIMARY KEY (user_id, specimen_id)
)
"""


@dataclass(frozen=True)
class GardenItem:
    specimen_id: str
    label_name: str
    saved_at: str  # ISO-8601 UTC timestamp


class SpecimenNotFoundError(ValueError):
    """Raised when a specimen_id has no known entry in the dataset metadata."""


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        is not None
    )


def _needs_schema_work(conn: sqlite3.Connection) -> bool:
    """Fast, lock-free check: is any create/migrate/recovery work outstanding?

    True when `garden` is missing, still on the old (`user_id`-less) schema, or
    a `garden_old` leftover from a previously aborted migration is present.
    """
    cols = conn.execute("PRAGMA table_info(garden)").fetchall()
    has_user_id = bool(cols) and "user_id" in {c[1] for c in cols}
    return (not has_user_id) or _table_exists(conn, "garden_old")


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Idempotent, crash-atomic, concurrency-safe migration for `garden.db`.

    Fresh DB → create the composite-PK schema. Old schema (`specimen_id` PK, no
    `user_id`) → rebuild and backfill every row as `DEFAULT_USER` so
    pre-scaffold data is preserved under the single-user bucket. Already
    migrated → no-op.

    The whole rebuild runs inside one `BEGIN IMMEDIATE` transaction so it is
    all-or-nothing: a crash or error mid-migration rolls back to the original
    table instead of stranding rows in `garden_old` (SQLite supports
    transactional DDL). The write lock is taken *before* re-checking the schema,
    so two threads hitting an unmigrated DB at once can't both rebuild — the one
    that loses the lock re-reads the finished schema and no-ops. A `garden_old`
    left by an older, non-atomic build is recovered (its rows are merged back)
    rather than silently ignored.
    """
    if not _needs_schema_work(conn):
        return  # common path: already migrated, no leftover — no lock needed
    # sqlite3's legacy isolation mode implicitly commits before DDL, which would
    # defeat atomicity; switch to manual control and drive the transaction here.
    prev_isolation = conn.isolation_level
    conn.isolation_level = None
    try:
        conn.execute("BEGIN IMMEDIATE")  # acquire the write lock, then re-check
        cols = conn.execute("PRAGMA table_info(garden)").fetchall()
        col_names = {c[1] for c in cols}
        leftover = _table_exists(conn, "garden_old")
        if not cols:
            conn.execute(_SCHEMA)
        elif "user_id" not in col_names:
            # Old schema → rebuild. Drop any stale leftover first (its data is a
            # duplicate of what's about to be renamed out of `garden`).
            conn.execute("DROP TABLE IF EXISTS garden_old")
            conn.execute("ALTER TABLE garden RENAME TO garden_old")
            conn.execute(_SCHEMA)
            conn.execute(
                "INSERT INTO garden (user_id, specimen_id, label_name, saved_at) "
                "SELECT ?, specimen_id, label_name, saved_at FROM garden_old",
                (DEFAULT_USER,),
            )
            conn.execute("DROP TABLE garden_old")
        elif leftover:
            # Recovery: a prior aborted (non-atomic) migration left the real
            # rows in `garden_old` beside a possibly-empty new `garden`. Merge
            # them back (PK-conflict rows are already present) and drop it.
            conn.execute(
                "INSERT OR IGNORE INTO garden (user_id, specimen_id, label_name, saved_at) "
                "SELECT ?, specimen_id, label_name, saved_at FROM garden_old",
                (DEFAULT_USER,),
            )
            conn.execute("DROP TABLE garden_old")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.isolation_level = prev_isolation


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    try:
        _ensure_schema(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def add_specimen(specimen_id: str, label_name: str, user_id: str = DEFAULT_USER) -> GardenItem:
    """Save (or refresh) a specimen in `user_id`'s garden. Caller must
    validate the specimen_id first via `resolve_label_name` — this function
    trusts its input. Different users may each save the same specimen_id
    independently (composite `(user_id, specimen_id)` primary key)."""
    saved_at = datetime.now(timezone.utc).isoformat()
    with _connection() as conn:
        conn.execute(
            "INSERT INTO garden (user_id, specimen_id, label_name, saved_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(user_id, specimen_id) DO UPDATE SET saved_at = excluded.saved_at",
            (user_id, specimen_id, label_name, saved_at),
        )
    return GardenItem(specimen_id=specimen_id, label_name=label_name, saved_at=saved_at)


def list_specimens(user_id: str = DEFAULT_USER) -> list[GardenItem]:
    """All of `user_id`'s saved specimens, most recently saved first."""
    with _connection() as conn:
        rows = conn.execute(
            "SELECT specimen_id, label_name, saved_at FROM garden WHERE user_id = ? "
            "ORDER BY saved_at DESC",
            (user_id,),
        ).fetchall()
    return [GardenItem(specimen_id=r[0], label_name=r[1], saved_at=r[2]) for r in rows]


def remove_specimen(specimen_id: str, user_id: str = DEFAULT_USER) -> bool:
    """Delete a specimen from `user_id`'s garden. Returns False if it wasn't
    saved by that user (idempotent; never affects other users' copies)."""
    with _connection() as conn:
        cur = conn.execute(
            "DELETE FROM garden WHERE user_id = ? AND specimen_id = ?", (user_id, specimen_id)
        )
        deleted = cur.rowcount > 0
    return deleted


# --------------------------------------------------------------------------- #
# Specimen validation — reuses the same embeddings-cache metadata
# search_service builds its gallery index from (read-only; never mutated).
# Reads metadata.json directly rather than `ml.embeddings.cache.load_embeddings`
# (which also loads the full embeddings.npz vector array) since only the
# id -> label_name mapping is needed to validate a save request.
# --------------------------------------------------------------------------- #
# Cache only a SUCCESSFULLY loaded catalog. A plain lru_cache would memoize the
# empty {} returned when metadata.json doesn't exist yet (e.g. the API started
# before the embeddings pipeline wrote it) — and then every later save would
# 404 for the process lifetime even after the file appears. Caching only the
# populated result lets a cold start recover on the next call.
_catalog: dict[str, str] | None = None


def _specimen_catalog() -> dict[str, str]:
    global _catalog
    if _catalog is not None:
        return _catalog
    meta_path = Path(settings.embeddings_cache_dir) / "metadata.json"
    if not meta_path.exists():
        return {}  # not built yet — do NOT cache the miss
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    _catalog = {sid: m["label_name"] for sid, m in metadata.get("specimens", {}).items()}
    return _catalog


def reset_specimen_catalog_cache() -> None:
    """Test helper: clears the cached specimen catalog singleton."""
    global _catalog
    _catalog = None


def resolve_label_name(specimen_id: str) -> str | None:
    """The specimen's label_name if it exists in the dataset metadata, else None."""
    return _specimen_catalog().get(specimen_id)
