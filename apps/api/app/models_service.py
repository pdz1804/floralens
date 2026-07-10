"""Support for `GET /api/models`: enumerate the ModelVersion artifacts under
`settings.models_dir` (each directory containing a `metadata.json` is one
version), mark which one is active (`settings.model_version`), and also expose
the synthetic "baseline" zero-shot backbone.

Read-only / best-effort, like `pipeline_service`: never loads torch or the
embedding backbone and never mutates anything — each version's key metrics are
read straight from its `metadata.json` (`val_metrics`), and the baseline's from
the on-disk baseline eval report if one exists. Any missing/unreadable file is
simply omitted, so the endpoint always returns a sensible response even before
the training/eval pipeline has produced its artifacts.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from apps.api.app.config import settings

logger = logging.getLogger(__name__)

_REPORTS_DIR = Path("ml/eval/reports")
# A small, stable subset of retrieval metrics surfaced per version (display
# only). Whatever of these a report/metadata actually carries is included.
_METRIC_KEYS = ("recall@1", "recall@5", "mrr", "map@5")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read %s: %s", path, exc)
        return None


def _key_metrics(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metrics:
        return None
    subset = {k: metrics[k] for k in _METRIC_KEYS if k in metrics}
    return subset or None


def _baseline_entry() -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": "baseline",
        "active": settings.model_version == "baseline",
    }
    # Best-effort backbone id from the embeddings-cache metadata (the frozen
    # backbone the baseline serves with); best-effort metrics from the baseline
    # eval report (prefer the held-out test split, fall back to val).
    emb_meta = _read_json(Path(settings.embeddings_cache_dir) / "metadata.json")
    if emb_meta and emb_meta.get("backbone"):
        entry["backbone"] = emb_meta["backbone"]
    report = _read_json(_REPORTS_DIR / "baseline_eval_report.json")
    if report is not None:
        metrics = _key_metrics(report.get("test") or report.get("val"))
        if metrics:
            entry["metrics"] = metrics
    return entry


def get_models() -> dict[str, Any]:
    """Return the available model versions and the active one:

        {
          "active": str,                       # settings.model_version
          "models": [
            {"name": str, "active": bool, "backbone"?: str, "metrics"?: dict},
            ...
          ],
        }

    The synthetic "baseline" entry is always first; every subdirectory of
    `settings.models_dir` that has a `metadata.json` follows, sorted by name.
    Never raises for a missing models dir/artifact — an environment without the
    trained artifacts yields just the baseline entry.
    """
    models: list[dict[str, Any]] = [_baseline_entry()]

    models_root = Path(settings.models_dir)
    if models_root.is_dir():
        for version_dir in sorted(models_root.iterdir()):
            if not version_dir.is_dir():
                continue
            metadata = _read_json(version_dir / "metadata.json")
            if metadata is None:
                continue  # not a ModelVersion artifact
            entry: dict[str, Any] = {
                "name": version_dir.name,
                "active": version_dir.name == settings.model_version,
            }
            backbone = metadata.get("base_model")
            if backbone:
                entry["backbone"] = backbone
            metrics = _key_metrics(metadata.get("val_metrics"))
            if metrics:
                entry["metrics"] = metrics
            models.append(entry)

    return {"active": settings.model_version, "models": models}
