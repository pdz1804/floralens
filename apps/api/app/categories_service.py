"""Support for `GET /api/categories`: the 102 Oxford-102 species aggregated
from the embeddings-cache metadata into one entry per species, each carrying a
representative gallery specimen (for a thumbnail) and the SAME stable per-species
color the Galaxy tab uses — so a species reads identically across both tabs.

Read-only/derived, like `galaxy_service` and the `/api/pipeline` support:
never re-embeds or mutates the gallery index. Sourced directly from
`metadata.json` (the id -> {label, label_name, split} mapping search_service
builds its gallery index from), not from the full embeddings vectors.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apps.api.app.config import settings
from apps.api.app.galaxy_service import _label_palette

# Cache only a SUCCESSFULLY built (non-empty) result. Mirrors
# garden_service._specimen_catalog: a plain lru_cache would memoize the empty
# list returned when metadata.json doesn't exist yet (API started before the
# embeddings pipeline wrote it) and then serve empty categories for the whole
# process lifetime even after the file appears. Caching only the populated
# result lets a cold start recover on the next call; a truly-missing file
# raises FileNotFoundError so the endpoint can surface a 503.
_categories: list[dict[str, Any]] | None = None


def get_categories() -> list[dict[str, Any]]:
    """One entry per distinct species, sorted by `label_name`. Each entry:
        {
          "label": int,               # numeric class id
          "label_name": str,          # species name
          "gallery_count": int,       # specimens with split == "gallery"
          "total_count": int,         # specimens across all splits
          "specimen_id": str,         # a representative gallery specimen id
          "color": str,               # stable per-species hex (== Galaxy tab)
        }

    Raises FileNotFoundError if metadata.json has not been built yet, so the
    endpoint can map it to a 503 (same contract as /api/galaxy)."""
    global _categories
    if _categories is not None:
        return _categories

    meta_path = Path(settings.embeddings_cache_dir) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"embeddings metadata not found at {meta_path}")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    specimens: dict[str, dict[str, Any]] = metadata.get("specimens", {})
    if not specimens:
        # Present but empty — do NOT cache; treat like a missing build.
        raise FileNotFoundError(f"embeddings metadata has no specimens: {meta_path}")

    # Aggregate per species. Track a representative gallery specimen id (first
    # one seen, id-sorted for stability so the thumbnail is deterministic).
    grouped: dict[str, dict[str, Any]] = {}
    for specimen_id in sorted(specimens):
        meta = specimens[specimen_id]
        name = meta["label_name"]
        entry = grouped.get(name)
        if entry is None:
            entry = grouped[name] = {
                "label": meta["label"],
                "label_name": name,
                "gallery_count": 0,
                "total_count": 0,
                "specimen_id": None,
            }
        entry["total_count"] += 1
        if meta["split"] == "gallery":
            entry["gallery_count"] += 1
            if entry["specimen_id"] is None:
                entry["specimen_id"] = specimen_id

    # Fall back to any specimen id if a species has no gallery split (keeps the
    # contract's specimen_id non-null so the thumbnail endpoint always resolves).
    palette = _label_palette(tuple(m["label_name"] for m in specimens.values()))
    result: list[dict[str, Any]] = []
    for name in sorted(grouped):
        entry = grouped[name]
        if entry["specimen_id"] is None:
            entry["specimen_id"] = next(
                sid for sid in sorted(specimens) if specimens[sid]["label_name"] == name
            )
        entry["color"] = palette[name]
        result.append(entry)

    _categories = result
    return _categories


def reset_categories_cache() -> None:
    """Test helper: clears the cached categories singleton."""
    global _categories
    _categories = None
