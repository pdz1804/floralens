"""Support for `GET /api/specimens/{specimen_id}`: richer detail for a single
specimen, sourced from the embeddings-cache metadata (the same id ->
{label, label_name, split, image_path} mapping search_service builds its
gallery index from).

Read-only / derived, like `categories_service`: never re-embeds or mutates the
gallery index. Raises FileNotFoundError when the cache metadata has not been
built yet (endpoint -> 503); returns None for an unknown id (endpoint -> 404).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apps.api.app.config import settings

# Cache only a SUCCESSFULLY loaded (non-empty) specimens map — same rationale as
# categories_service: a plain lru_cache would memoize the empty result from a
# not-yet-built cache and never recover. A truly-missing/empty file raises
# FileNotFoundError so the endpoint surfaces a clean 503 and a later call can
# still recover once the pipeline has written the file.
_specimens: dict[str, dict[str, Any]] | None = None


def _load_specimens() -> dict[str, dict[str, Any]]:
    global _specimens
    if _specimens is not None:
        return _specimens

    meta_path = Path(settings.embeddings_cache_dir) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"embeddings metadata not found at {meta_path}")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    specimens: dict[str, dict[str, Any]] = metadata.get("specimens", {})
    if not specimens:
        # Present but empty — do NOT cache; treat like a missing build.
        raise FileNotFoundError(f"embeddings metadata has no specimens: {meta_path}")

    _specimens = specimens
    return _specimens


def get_specimen_detail(specimen_id: str) -> dict[str, Any] | None:
    """Return one specimen's detail, or None if the id is unknown:

        {
          "specimen_id": str,
          "label": int,
          "label_name": str,
          "split": str,                 # gallery | val | test
          "image_url": str,             # "/api/specimen/{id}/image"
        }

    Raises FileNotFoundError if the embeddings cache has not been built yet, so
    the endpoint can map it to a 503 (same contract as /api/categories)."""
    meta = _load_specimens().get(specimen_id)
    if meta is None:
        return None
    return {
        "specimen_id": specimen_id,
        "label": meta["label"],
        "label_name": meta["label_name"],
        "split": meta["split"],
        "image_url": f"/api/specimen/{specimen_id}/image",
    }


def reset_specimen_detail_cache() -> None:
    """Test helper: clears the cached specimens map singleton."""
    global _specimens
    _specimens = None
