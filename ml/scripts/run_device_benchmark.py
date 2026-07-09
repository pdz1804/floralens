"""CLI entry point: GPU-vs-CPU device benchmark (task deliverable "GPU-vs-CPU
benchmark"). Times:
  1. Embedding a fixed batch of real (preprocessed) gallery images through the
     active backbone, once on CPU and once on CUDA (if a CUDA device exists).
  2. One short projection-head training epoch over the cached augmented train
     embeddings, on the same two devices — the head is tiny (a linear/MLP
     layer), so this also reports whether GPU training pays off for a model
     this small, honestly, rather than assuming it does.

Writes `ml/eval/reports/device_benchmark.json` and logs the same numbers to
MLflow. Real measurements only — every number here comes from an actual timed
run, never estimated or fabricated.

Usage:
    venv/Scripts/python.exe -m ml.scripts.run_device_benchmark
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from ml.data.splits import load_manifest
from ml.device import DEVICE_ENV_VAR
from ml.embeddings import backbone as backbone_module
from ml.preprocess.pipeline import preprocess
from ml.tracking.mlflow_utils import mlflow_run
from ml.train.trainer import TrainConfig, train_head

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BATCH_SIZE = 64
TRAIN_CACHE_DIR = "ml/data/embeddings_cache/train"
REPORT_PATH = Path("ml/eval/reports/device_benchmark.json")


def _sample_preprocessed_images(n: int = BATCH_SIZE) -> list[Image.Image]:
    """A deterministic (sorted-by-id) sample of real, already-preprocessed
    gallery images — so the benchmark measures backbone forward-pass time
    only, not preprocessing time."""
    manifest = load_manifest()
    gallery = sorted(
        (r for r in manifest["records"] if r["split"] == "gallery"), key=lambda r: r["id"]
    )[:n]
    images = []
    for rec in gallery:
        with Image.open(rec["image_path"]) as img:
            img.load()
            clean, _steps = preprocess(img)
            images.append(clean.copy())
    return images


def _benchmark_embed(images: list[Image.Image], device_pref: str) -> dict[str, Any]:
    os.environ[DEVICE_ENV_VAR] = device_pref
    backbone_module.reset_backbone_cache()
    # Warm-up: pays model load + (for cuda) context/kernel init cost once,
    # excluded from the timed loop below.
    backbone_module.embed_image(images[0])
    if device_pref == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for img in images:
        backbone_module.embed_image(img)
    if device_pref == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    return {
        "requested_device": device_pref,
        "resolved_device": backbone_module.device_str(),
        "backbone": backbone_module.backbone_name(),
        "num_images": len(images),
        "elapsed_seconds": elapsed,
        "images_per_second": (len(images) / elapsed) if elapsed > 0 else None,
    }


def _load_train_cache() -> tuple[Any, Any, int]:
    import numpy as np

    from ml.embeddings.cache import load_embeddings

    vectors, metadata = load_embeddings(TRAIN_CACHE_DIR)
    specimens = metadata["specimens"]
    ids = sorted(vectors.keys())
    embeddings = np.stack([vectors[i] for i in ids], axis=0)
    labels = np.array([specimens[i]["label"] for i in ids], dtype=np.int64)
    num_classes = len(set(specimens[i]["label"] for i in ids))
    return embeddings, labels, num_classes


def _benchmark_train_epoch(device_pref: str) -> dict[str, Any] | None:
    """One short training run (single epoch, no early stopping) timed on
    `device_pref`. Returns None if the required caches are not present yet
    (e.g. this benchmark is run before the finetune sweep's inputs exist)."""
    from ml.data.splits import load_manifest
    from ml.embeddings.backbone import embedding_dim
    from ml.embeddings.cache import load_embeddings
    from ml.eval.harness import EmbeddedSpecimen

    embeddings_cache = Path("ml/data/embeddings_cache/embeddings.npz")
    train_cache = Path(TRAIN_CACHE_DIR) / "embeddings.npz"
    if not embeddings_cache.exists() or not train_cache.exists():
        logger.warning("train-epoch benchmark skipped: embeddings caches not built yet")
        return None

    manifest = load_manifest()
    vectors, metadata = load_embeddings()
    specimens = metadata["specimens"]
    gallery_specimens, val_specimens = [], []
    for specimen_id, vector in vectors.items():
        meta = specimens[specimen_id]
        item = EmbeddedSpecimen(id=specimen_id, label=meta["label"], label_name=meta["label_name"], vector=vector)
        if meta["split"] == "gallery":
            gallery_specimens.append(item)
        elif meta["split"] == "val":
            val_specimens.append(item)

    train_embeddings, train_labels, num_classes = _load_train_cache()
    config = TrainConfig(
        loss="arcface", lr=1e-3, head_hidden_dim=0, output_dim=256, margin=0.3, scale=30.0,
        batch_size=128, max_epochs=1, early_stop_patience=1, seed=42,
    )

    device = torch.device(device_pref)
    if device_pref == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    train_head(
        config, train_embeddings, train_labels, num_classes,
        gallery_specimens, val_specimens, embedding_dim(), device=device,
    )
    if device_pref == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    return {
        "device": device_pref,
        "num_train_views": int(train_embeddings.shape[0]),
        "num_gallery": len(gallery_specimens),
        "num_val": len(val_specimens),
        "elapsed_seconds": elapsed,
        "note": "single epoch, includes the val-side retrieval eval (CPU/numpy, "
        "not device-dependent) used for early stopping — not pure GPU compute time",
    }


def main() -> None:
    devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    if not torch.cuda.is_available():
        logger.warning("no CUDA device available; benchmark will only cover cpu")

    images = _sample_preprocessed_images()
    logger.info("benchmarking embed throughput over %d real gallery images", len(images))

    embed_results: dict[str, Any] = {}
    for device_pref in devices:
        result = _benchmark_embed(images, device_pref)
        embed_results[device_pref] = result
        logger.info(
            "embed[%s]: %.2f img/s (%.3fs for %d images)",
            device_pref, result["images_per_second"], result["elapsed_seconds"], result["num_images"],
        )

    train_results: dict[str, Any] = {}
    for device_pref in devices:
        result = _benchmark_train_epoch(device_pref)
        if result is not None:
            train_results[device_pref] = result
            logger.info("train_epoch[%s]: %.3fs", device_pref, result["elapsed_seconds"])

    embed_speedup = None
    if "cuda" in embed_results and embed_results["cpu"]["elapsed_seconds"] > 0:
        embed_speedup = embed_results["cpu"]["elapsed_seconds"] / embed_results["cuda"]["elapsed_seconds"]

    train_speedup = None
    if "cuda" in train_results and train_results.get("cpu", {}).get("elapsed_seconds", 0) > 0:
        train_speedup = train_results["cpu"]["elapsed_seconds"] / train_results["cuda"]["elapsed_seconds"]

    report = {
        "batch_size": len(images),
        "embed": embed_results,
        "embed_speedup_cuda_over_cpu": embed_speedup,
        "train_epoch": train_results,
        "train_epoch_speedup_cuda_over_cpu": train_speedup,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("device benchmark report saved to %s", REPORT_PATH)

    with mlflow_run(
        "device_benchmark", tags={"stage": "device_benchmark"}, params={"batch_size": len(images)}
    ) as mlf:
        for device_pref, result in embed_results.items():
            mlf.log_metric(f"embed_{device_pref}_images_per_second", result["images_per_second"])
            mlf.log_metric(f"embed_{device_pref}_elapsed_seconds", result["elapsed_seconds"])
        if embed_speedup is not None:
            mlf.log_metric("embed_speedup_cuda_over_cpu", embed_speedup)
        for device_pref, result in train_results.items():
            mlf.log_metric(f"train_epoch_{device_pref}_elapsed_seconds", result["elapsed_seconds"])
        if train_speedup is not None:
            mlf.log_metric("train_epoch_speedup_cuda_over_cpu", train_speedup)
        mlf.log_artifact(str(REPORT_PATH))

    # Restore the default device-resolution behavior for any subsequent
    # script run in this same process (this script's benchmarking loop forces
    # FLORALENS_DEVICE per iteration; a following import should see "auto").
    os.environ.pop(DEVICE_ENV_VAR, None)
    backbone_module.reset_backbone_cache()


if __name__ == "__main__":
    main()
