"""Garden service: persists the user's saved specimens ("My Garden", PRD
Phase 7 / `GardenPlant`) so the collection survives an API restart.

**Storage choice:** SQLite (stdlib `sqlite3`, no new dependency) in a single
file under `GARDEN_DB_PATH` (default `apps/api/data/garden.db`). A flat JSON
file would also work for this small, single-table shape, but SQLite gives
concurrency-safe writes (FastAPI's threadpool can run request handlers on
multiple threads) and an upsert-on-save via `ON CONFLICT`, both of which a
hand-rolled JSON read/modify/write would need to reimplement. The DB file is
created lazily on first use and is gitignored (runtime state, not source).

One row per `specimen_id` — saving an already-saved specimen just refreshes
`saved_at` rather than erroring or duplicating.
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

from apps.api.app.config import settings

# Mutable at module scope (not a frozen default-arg) so tests can repoint it
# to a tmp_path via `monkeypatch.setattr(garden_service, "_DB_PATH", ...)`,
# matching this codebase's existing `monkeypatch.setattr(search_service, ...)`
# test convention rather than adding a bespoke setter API.
_DB_PATH = Path(os.environ.get("GARDEN_DB_PATH", "apps/api/data/garden.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS garden (
    specimen_id TEXT PRIMARY KEY,
    label_name TEXT NOT NULL,
    saved_at TEXT NOT NULL
)
"""


@dataclass(frozen=True)
class GardenItem:
    specimen_id: str
    label_name: str
    saved_at: str  # ISO-8601 UTC timestamp


class SpecimenNotFoundError(ValueError):
    """Raised when a specimen_id has no known entry in the dataset metadata."""


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def add_specimen(specimen_id: str, label_name: str) -> GardenItem:
    """Save (or refresh) a specimen in the garden. Caller must validate the
    specimen_id first via `resolve_label_name` — this function trusts its input."""
    saved_at = datetime.now(timezone.utc).isoformat()
    with _connection() as conn:
        conn.execute(
            "INSERT INTO garden (specimen_id, label_name, saved_at) VALUES (?, ?, ?) "
            "ON CONFLICT(specimen_id) DO UPDATE SET saved_at = excluded.saved_at",
            (specimen_id, label_name, saved_at),
        )
    return GardenItem(specimen_id=specimen_id, label_name=label_name, saved_at=saved_at)


def list_specimens() -> list[GardenItem]:
    """All saved specimens, most recently saved first."""
    with _connection() as conn:
        rows = conn.execute(
            "SELECT specimen_id, label_name, saved_at FROM garden ORDER BY saved_at DESC"
        ).fetchall()
    return [GardenItem(specimen_id=r[0], label_name=r[1], saved_at=r[2]) for r in rows]


def remove_specimen(specimen_id: str) -> bool:
    """Delete a saved specimen. Returns False if it wasn't saved (idempotent)."""
    with _connection() as conn:
        cur = conn.execute("DELETE FROM garden WHERE specimen_id = ?", (specimen_id,))
        deleted = cur.rowcount > 0
    return deleted


# --------------------------------------------------------------------------- #
# Specimen validation — reuses the same embeddings-cache metadata
# search_service builds its gallery index from (read-only; never mutated).
# Reads metadata.json directly rather than `ml.embeddings.cache.load_embeddings`
# (which also loads the full embeddings.npz vector array) since only the
# id -> label_name mapping is needed to validate a save request.
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _specimen_catalog() -> dict[str, str]:
    meta_path = Path(settings.embeddings_cache_dir) / "metadata.json"
    if not meta_path.exists():
        return {}
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return {sid: m["label_name"] for sid, m in metadata.get("specimens", {}).items()}


def reset_specimen_catalog_cache() -> None:
    """Test helper: clears the cached specimen catalog singleton."""
    _specimen_catalog.cache_clear()


def resolve_label_name(specimen_id: str) -> str | None:
    """The specimen's label_name if it exists in the dataset metadata, else None."""
    return _specimen_catalog().get(specimen_id)
