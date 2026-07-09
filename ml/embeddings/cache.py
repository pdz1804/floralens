"""Embed a list of manifest records and persist vectors + metadata to disk.

Output layout under `out_dir`:
  embeddings.npz   -> arrays keyed by specimen id ("flowers102-00123" -> vector)
  metadata.json    -> {id: {label, label_name, split, image_path}, model_version, backbone}
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ml.embeddings.backbone import backbone_name, embed_image
from ml.preprocess.pipeline import preprocess_fingerprint
from ml.preprocess.pipeline import preprocess

logger = logging.getLogger(__name__)


def embed_records(
    records: list[dict[str, Any]], model_version: str = "baseline"
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Embed each record's image. Returns (id -> vector, id -> metadata).

    Every image runs through the deterministic CV preprocessing pipeline
    (`ml.preprocess.pipeline.preprocess`: EXIF auto-orient, RGB, resize,
    center crop, white balance, CLAHE) before reaching the backbone, so the
    gallery/val/test embedding cache is built under the exact same
    preprocessing the live `/api/search` query path applies.
    """
    vectors: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    start = time.time()
    for i, rec in enumerate(records):
        with Image.open(rec["image_path"]) as img:
            img.load()
            clean, _steps = preprocess(img)
            vectors[rec["id"]] = embed_image(clean)
        metadata[rec["id"]] = {
            "label": rec["label"],
            "label_name": rec["label_name"],
            "split": rec["split"],
            "image_path": rec["image_path"],
        }
        if (i + 1) % 200 == 0:
            logger.info("embedded %d/%d images (%.1fs elapsed)", i + 1, len(records), time.time() - start)
    logger.info(
        "embedded %d images with %s in %.1fs", len(records), backbone_name(), time.time() - start
    )
    return vectors, {
        "model_version": model_version,
        "backbone": backbone_name(),
        "preprocess_fingerprint": preprocess_fingerprint(),
        "specimens": metadata,
    }


def save_embeddings(
    vectors: dict[str, np.ndarray], metadata: dict[str, Any], out_dir: str = "ml/data/embeddings_cache"
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    np.savez(out_path / "embeddings.npz", **vectors)
    (out_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_embeddings(
    in_dir: str = "ml/data/embeddings_cache",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    in_path = Path(in_dir)
    npz = np.load(in_path / "embeddings.npz")
    vectors = {key: npz[key] for key in npz.files}
    metadata = json.loads((in_path / "metadata.json").read_text(encoding="utf-8"))
    return vectors, metadata
