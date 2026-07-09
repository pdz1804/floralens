"""Search service: builds the gallery vector index from the cached embeddings
and exposes `search_image(pil_image, top_k)` used by the /api/search route.
"""
from __future__ import annotations

import functools
import io
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from apps.api.app.config import settings
from ml.embeddings.backbone import embed_image
from ml.embeddings.cache import load_embeddings
from ml.eval.calibration import confidence_band
from ml.index.vector_store import VectorStore

logger = logging.getLogger(__name__)


class SearchResultItem:
    __slots__ = ("specimen_id", "label", "label_name", "score", "confidence", "band")

    def __init__(
        self,
        specimen_id: str,
        label: int,
        label_name: str,
        score: float,
        confidence: float | None = None,
        band: str | None = None,
    ) -> None:
        self.specimen_id = specimen_id
        self.label = label
        self.label_name = label_name
        self.score = score
        # Calibrated confidence (0..1) + high/medium/low band (PRD §14.6).
        # Both stay None on the "baseline" model version (raw cosine, no
        # calibrator fit yet) — additive/optional so the existing
        # specimen_id/label/label_name/score contract is unchanged.
        self.confidence = confidence
        self.band = band


@functools.lru_cache(maxsize=1)
def _load_candidate(model_version: str) -> tuple[Any, dict, Any | None] | None:
    """Load a promoted candidate's projection head (+ optional calibrator).

    Returns None for "baseline" or if no ModelVersion artifact directory
    exists for `model_version` (e.g. an unset/typo'd env var) — callers then
    fall back to the frozen backbone with no projection, matching Phase 0-1
    behavior exactly.
    """
    if model_version == "baseline":
        return None
    candidate_dir = Path(settings.models_dir) / model_version
    if not (candidate_dir / "head.pt").exists():
        logger.warning("MODEL_VERSION=%s has no artifact at %s; falling back to baseline", model_version, candidate_dir)
        return None

    from ml.train.model_io import load_candidate_head  # local import: keeps torch off the baseline hot path

    head, metadata = load_candidate_head(candidate_dir)
    calibrator = None
    calibrator_path = candidate_dir / "calibrator.pkl"
    if calibrator_path.exists():
        calibrator = pickle.loads(calibrator_path.read_bytes())
    return head, metadata, calibrator


def reset_candidate_cache() -> None:
    """Test helper: clears the cached candidate head/calibrator singleton."""
    _load_candidate.cache_clear()


@functools.lru_cache(maxsize=1)
def get_gallery_store() -> VectorStore:
    """Build (once) the in-memory gallery vector store from the embeddings cache.

    Only specimens tagged split=="gallery" are indexed, matching the PRD
    rule that production search never queries against held-out val/test
    query partitions (§14.7). If `settings.model_version` names a promoted
    candidate, gallery vectors are projected through its head before
    indexing so query-time and index-time embeddings live in the same space.
    """
    vectors, metadata = load_embeddings(settings.embeddings_cache_dir)
    candidate = _load_candidate(settings.model_version)
    model_version = settings.model_version if candidate else metadata.get("model_version", "baseline")

    store = VectorStore(model_version=model_version)
    specimens = metadata["specimens"]
    gallery_ids = [sid for sid, meta in specimens.items() if meta["split"] == "gallery"]

    if candidate:
        head, _, _ = candidate
        from ml.train.model_io import project_embeddings

        raw = np.stack([vectors[sid] for sid in gallery_ids], axis=0)
        projected = project_embeddings(head, raw)
    else:
        projected = np.stack([vectors[sid] for sid in gallery_ids], axis=0)

    for specimen_id, vector in zip(gallery_ids, projected):
        meta = specimens[specimen_id]
        store.add(
            specimen_id,
            vector,
            {
                "label": meta["label"],
                "label_name": meta["label_name"],
                "image_path": meta["image_path"],
            },
        )
    logger.info("gallery vector store ready: %d specimens (model_version=%s)", len(gallery_ids), model_version)
    return store


def reset_gallery_store_cache() -> None:
    """Test helper: clears the cached gallery store singleton."""
    get_gallery_store.cache_clear()


# Base dir all specimen images must live under — a resolved-path allowlist so a
# crafted specimen_id can never escape the dataset directory (defense in depth on
# top of the metadata id lookup).
_IMAGE_BASE = Path("ml/data/raw").resolve()


@functools.lru_cache(maxsize=1)
def _specimen_image_index() -> dict[str, Path]:
    """Map every known specimen_id -> its on-disk image path (validated once).

    Only ids present in the dataset metadata resolve, and only paths that stay
    within `_IMAGE_BASE` are kept — so `/api/specimen/{id}/image` cannot be used
    for path traversal.
    """
    _, metadata = load_embeddings(settings.embeddings_cache_dir)
    index: dict[str, Path] = {}
    for sid, meta in metadata["specimens"].items():
        path = Path(meta["image_path"]).resolve()
        try:
            path.relative_to(_IMAGE_BASE)
        except ValueError:
            continue  # outside the dataset dir — skip defensively
        index[sid] = path
    return index


def reset_specimen_image_index_cache() -> None:
    """Test helper: clears the cached specimen-image index singleton."""
    _specimen_image_index.cache_clear()


def get_specimen_image_path(specimen_id: str) -> Path | None:
    """Resolve a specimen_id to its image file, or None if unknown/missing."""
    path = _specimen_image_index().get(specimen_id)
    if path is None or not path.exists():
        return None
    return path


def strip_exif_and_load(image_bytes: bytes) -> Image.Image:
    """Decode image bytes and return a clean RGB copy with EXIF metadata dropped."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img.load()
        clean = Image.new("RGB", img.size)
        clean.paste(img.convert("RGB"))
        return clean


def search_image(image: Image.Image, top_k: int | None = None) -> tuple[str, list[SearchResultItem]]:
    """Embed the query image and return (model_version, ranked results).

    On "baseline" (default), behavior is unchanged from Phase 0-1: raw
    cosine score, no confidence/band. If a promoted candidate is active
    (settings.model_version), the query embedding is projected through its
    head to match the gallery's space, and — if a calibrator artifact was
    persisted for that candidate — each result also carries a calibrated
    `confidence` (0..1) and `band` (high/medium/low), per PRD §14.6.
    """
    top_k = top_k or settings.default_top_k
    store = get_gallery_store()
    vector: np.ndarray = embed_image(image)

    candidate = _load_candidate(settings.model_version)
    calibrator = None
    if candidate:
        head, _, calibrator = candidate
        from ml.train.model_io import project_embeddings

        vector = project_embeddings(head, vector.reshape(1, -1))[0]

    results = store.query(vector, top_k=top_k)
    items = []
    for r in results:
        confidence = None
        band = None
        if calibrator is not None:
            confidence = float(calibrator.predict(np.array([r.score]))[0])
            band = confidence_band(confidence)
        items.append(
            SearchResultItem(
                specimen_id=r.id,
                label=r.metadata["label"],
                label_name=r.metadata["label_name"],
                score=r.score,
                confidence=confidence,
                band=band,
            )
        )
    return store.model_version, items
