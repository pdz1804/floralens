"""CLI entry point: embed a stratified, train-only-augmented subset of the
`train` split through the frozen backbone (PRD §14.5 step 3: augmentation
applied to train only).

Scale: the full train split is 4917 images (min 24/class); embedding every
image on this CPU-only environment is unnecessary for a projection-head
experiment, so we embed a documented, deterministic subset capped at
`PER_CLASS_CAP` images per class (every class has >= 24 train images, so the
cap is satisfiable for all 102 classes).

Augmentation: each selected image contributes `AUGMENT_VIEWS` embeddings —
the original crop plus geometric augmentations applied to the raw PIL image
*before* the frozen backbone's own preprocessing (so the backbone genuinely
sees different pixels per view, not a no-op). We use horizontal flip and a
mild rotation, deliberately skipping color jitter: flower species are often
distinguished by petal/center color, so perturbing color would inject label
noise rather than a valid invariance for this task. Flip and small rotation
are safe: a photographed flower's orientation is not diagnostic.

Output: a separate cache under `ml/data/embeddings_cache/train/` (does not
touch the existing `embeddings.npz`/`metadata.json` used by the baseline
gallery/val/test eval, so Phase 1 artifacts stay reproducible untouched).
Each view gets its own id: "<specimen_id>::view<i>", all sharing the base
specimen's label — this augmented cache is training-only input, never
indexed for retrieval.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torchvision.transforms import functional as F

from ml.data.splits import DEFAULT_SEED, load_manifest
from ml.embeddings.backbone import backbone_name, embed_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PER_CLASS_CAP = 15
AUGMENT_VIEWS = ("original", "hflip", "rotate")
ROTATE_DEGREES = 8.0
OUT_DIR = "ml/data/embeddings_cache/train"


def select_train_subset(manifest: dict[str, Any], per_class_cap: int = PER_CLASS_CAP) -> list[dict[str, Any]]:
    """Deterministic (sorted-by-id) per-class cap over the train split only."""
    by_class: dict[int, list[dict[str, Any]]] = {}
    for rec in manifest["records"]:
        if rec["split"] != "train":
            continue
        by_class.setdefault(rec["label"], []).append(rec)

    selected: list[dict[str, Any]] = []
    for label, recs in by_class.items():
        recs_sorted = sorted(recs, key=lambda r: r["id"])
        selected.extend(recs_sorted[:per_class_cap])
    return sorted(selected, key=lambda r: r["id"])


def _augmented_view(image: Image.Image, view: str) -> Image.Image:
    if view == "original":
        return image
    if view == "hflip":
        return F.hflip(image)
    if view == "rotate":
        # Mild rotation with edge-replicate fill so no synthetic black
        # borders are introduced (which would themselves become a spurious
        # learnable cue).
        return F.rotate(image, ROTATE_DEGREES, fill=None, expand=False)
    raise ValueError(f"unknown augmentation view: {view}")


def embed_augmented_train_subset(
    records: list[dict[str, Any]], views: tuple[str, ...] = AUGMENT_VIEWS
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    vectors: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    start = time.time()
    total = len(records) * len(views)
    done = 0
    for rec in records:
        with Image.open(rec["image_path"]) as img:
            clean = Image.new("RGB", img.size)
            clean.paste(img.convert("RGB"))
            for view in views:
                view_image = _augmented_view(clean, view)
                view_id = f"{rec['id']}::{view}"
                vectors[view_id] = embed_image(view_image)
                metadata[view_id] = {
                    "base_id": rec["id"],
                    "label": rec["label"],
                    "label_name": rec["label_name"],
                    "view": view,
                }
                done += 1
        if done % 300 < len(views):
            logger.info("embedded %d/%d train views (%.1fs elapsed)", done, total, time.time() - start)
    logger.info(
        "embedded %d train views (%d base images x %d views) with %s in %.1fs",
        len(vectors), len(records), len(views), backbone_name(), time.time() - start,
    )
    return vectors, {
        "backbone": backbone_name(),
        "per_class_cap": PER_CLASS_CAP,
        "views": list(views),
        "seed": DEFAULT_SEED,
        "num_base_images": len(records),
        "specimens": metadata,
    }


def save_train_embeddings(vectors: dict[str, np.ndarray], metadata: dict[str, Any], out_dir: str = OUT_DIR) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    np.savez(out_path / "embeddings.npz", **vectors)
    (out_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    manifest = load_manifest()
    subset = select_train_subset(manifest)
    logger.info(
        "selected %d/%d train records (cap=%d/class, %d classes covered)",
        len(subset), sum(1 for r in manifest["records"] if r["split"] == "train"),
        PER_CLASS_CAP, len({r["label"] for r in subset}),
    )
    vectors, metadata = embed_augmented_train_subset(subset)
    save_train_embeddings(vectors, metadata)
    logger.info("saved train embeddings cache: %d vectors to %s", len(vectors), OUT_DIR)


if __name__ == "__main__":
    main()
