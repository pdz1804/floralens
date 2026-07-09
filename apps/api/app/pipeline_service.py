"""Support services for the ML-pipeline transparency endpoints:
`POST /api/preprocess-preview` and `GET /api/pipeline` (surfaced by the web
app's Pipeline page). Both are read-only/derived — they never mutate the
gallery index or trigger training.

`/api/pipeline` is sourced entirely from artifacts already written to disk
by the training/eval scripts (split manifest, embeddings cache metadata,
candidate eval/calibration reports, promotion decision, model metadata) —
never by loading the backbone or re-running eval, so the endpoint stays
cheap and reflects exactly what was actually measured.
"""
from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image

from apps.api.app.config import settings
from apps.api.app.search_service import decode_image_bytes
from ml.preprocess.pipeline import preprocess, preprocess_steps_only

logger = logging.getLogger(__name__)

_PREVIEW_MAX_SIDE = 512  # cap before/after thumbnail size for a small response payload
_MANIFEST_PATH = Path("ml/data/manifests/split_manifest.json")
_EMBEDDINGS_METADATA_PATH = Path("ml/data/embeddings_cache/metadata.json")
_REPORTS_DIR = Path("ml/eval/reports")


def _to_base64_png(image: Image.Image, max_side: int = _PREVIEW_MAX_SIDE) -> str:
    img = image
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        new_size = (max(1, round(img.size[0] * scale)), max(1, round(img.size[1] * scale)))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_preprocess_preview(image_bytes: bytes) -> dict[str, Any]:
    """Run preprocessing on an uploaded image and return steps + before/after
    thumbnails as base64 PNG, for the UI's before/after preview."""
    before = decode_image_bytes(image_bytes)
    after, steps = preprocess(before)
    return {
        "steps": steps,
        "before_png_b64": _to_base64_png(before),
        "after_png_b64": _to_base64_png(after),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read %s: %s", path, exc)
        return None


def _dataset_snapshot() -> dict[str, Any]:
    manifest = _read_json(_MANIFEST_PATH)
    embeddings_meta = _read_json(_EMBEDDINGS_METADATA_PATH)

    if embeddings_meta is not None:
        split_counts: dict[str, int] = {}
        for meta in embeddings_meta.get("specimens", {}).values():
            split_counts[meta["split"]] = split_counts.get(meta["split"], 0) + 1
        total = sum(split_counts.values())
    else:
        split_counts, total = {}, 0

    return {
        "name": manifest["dataset"] if manifest else "oxford-102-flowers",
        "total": total,
        "classes": manifest["num_classes"] if manifest else None,
        "splits": split_counts,
    }


def _active_model_metadata() -> dict[str, Any] | None:
    if settings.model_version == "baseline":
        return None
    return _read_json(Path(settings.models_dir) / settings.model_version / "metadata.json")


def _backbone_snapshot(model_metadata: dict[str, Any] | None) -> dict[str, Any]:
    embeddings_meta = _read_json(_EMBEDDINGS_METADATA_PATH)
    backbone_id = (model_metadata or {}).get("base_model") or (embeddings_meta or {}).get(
        "backbone", "unknown"
    )
    dim = (model_metadata or {}).get("input_dim")
    return {"name": backbone_id, "version": backbone_id, "dim": dim, "frozen": True}


def _eval_snapshot() -> dict[str, Any]:
    # Prefer the one-shot candidate test report (val + test, same protocol);
    # fall back to the zero-shot baseline report if no candidate has run yet.
    candidate = _read_json(_REPORTS_DIR / "candidate_test_eval_report.json")
    report = candidate or _read_json(_REPORTS_DIR / "baseline_eval_report.json")
    if report is None:
        return {"val": None, "test": None}
    return {"val": report.get("val"), "test": report.get("test")}


def _calibration_snapshot() -> dict[str, Any]:
    report = _read_json(_REPORTS_DIR / "candidate_calibration_report.json")
    if report is None:
        return {"ece_before": None, "ece_after": None, "bands": None}
    return {
        "ece_before": report.get("ece_test_before_calibration"),
        "ece_after": report.get("ece_test_after_calibration"),
        "bands": report.get("test_reliability_bins"),
    }


def _promotion_snapshot() -> dict[str, Any]:
    report = _read_json(_REPORTS_DIR / "promotion_decision.json")
    if report is None:
        return {"decision": None, "reason": None}
    reasons = report.get("reasons") or []
    return {"decision": report.get("decision"), "reason": "; ".join(reasons) if reasons else None}


def get_pipeline_snapshot() -> dict[str, Any]:
    """Assemble the full `/api/pipeline` payload from on-disk artifacts."""
    model_metadata = _active_model_metadata()
    return {
        "dataset": _dataset_snapshot(),
        "preprocessing": preprocess_steps_only(),
        "backbone": _backbone_snapshot(model_metadata),
        "eval": _eval_snapshot(),
        "calibration": _calibration_snapshot(),
        "promotion": _promotion_snapshot(),
        "model_version": settings.model_version,
    }
