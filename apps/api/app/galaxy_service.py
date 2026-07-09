"""Support for `GET /api/galaxy`: the gallery's precomputed 3D embedding
projection (see `ml/scripts/build_galaxy_projection.py`), enriched with a
stable per-species color for the Galaxy tab's point-cloud viewer.

Read-only/derived, like the `/api/pipeline` support in `pipeline_service.py`:
never re-embeds or mutates the gallery index. The projection JSON is built
once (lazily, on first call) if missing, then served from an in-memory cache.
"""
from __future__ import annotations

import colorsys
import functools
from typing import Any

from apps.api.app.config import settings
from ml.scripts.build_galaxy_projection import load_or_build


def _hex_color(hue: float) -> str:
    r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.55)
    return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))


@functools.lru_cache(maxsize=1)
def _label_palette(label_names: tuple[str, ...]) -> dict[str, str]:
    """Stable hue-wheel color per distinct species name.

    Ordered alphabetically (not by first-seen order in the projection file)
    so a given species always gets the same color across rebuilds/requests.
    """
    ordered = sorted(set(label_names))
    n = len(ordered) or 1
    return {name: _hex_color(i / n) for i, name in enumerate(ordered)}


@functools.lru_cache(maxsize=1)
def get_galaxy_points() -> list[dict[str, Any]]:
    """Load (building once if missing) the gallery's 3D projection, with each
    point's stable per-species color attached, for the Galaxy tab."""
    points = load_or_build(settings.embeddings_cache_dir)
    palette = _label_palette(tuple(p["label_name"] for p in points))
    return [{**p, "color": palette[p["label_name"]]} for p in points]


def reset_galaxy_cache() -> None:
    """Test helper: clears the cached galaxy points/palette singletons."""
    get_galaxy_points.cache_clear()
    _label_palette.cache_clear()
