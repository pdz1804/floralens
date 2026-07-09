"""Species description lookup (DISPLAY ONLY).

Loads `flower_descriptions.json` — a curated, factual, 1-2 sentence
botanical description per Oxford-102 `label_name` — and exposes it by name.
This is presentation metadata for the `/api/search` result cards; it is
never read by anything in `ml/embeddings`, `ml/train`, or `ml/eval` and must
never influence embeddings, training, or evaluation.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

_DESCRIPTIONS_PATH = Path(__file__).parent / "flower_descriptions.json"


@functools.lru_cache(maxsize=1)
def _load() -> dict[str, str]:
    return json.loads(_DESCRIPTIONS_PATH.read_text(encoding="utf-8"))


def get_description(label_name: str) -> str | None:
    """Return the curated description for a species label_name, or None if
    the name is not in the curated set (never raises on unknown input)."""
    return _load().get(label_name)


def reset_description_cache() -> None:
    """Test helper: clears the cached description map singleton."""
    _load.cache_clear()
