"""CLI entry point: Phase 3 fine-tuning sweep — ArcFace hyperparameter grid
(+ one triplet comparison run), early stopping and model selection on
**val only** (PRD §14.5, §18.3). The test split is never read here; every id
touched is asserted against `ml.train.isolation_guard` before training.

Usage:
    venv/Scripts/python.exe -m ml.scripts.run_finetune_sweep

Inputs:
  - ml/data/manifests/split_manifest.json          (train/val/test/gallery ids)
  - ml/data/embeddings_cache/{embeddings.npz,metadata.json}   (gallery+val+test, backbone)
  - ml/data/embeddings_cache/train/{embeddings.npz,metadata.json}  (augmented train subset, backbone)

Outputs:
  - ml/eval/reports/experiments/<run_id>.json   (per-run config + curves + val metrics + ids used)
  - ml/eval/reports/finetune_sweep_summary.json (all runs ranked by val Recall@5)
  - ml/models/<CANDIDATE_VERSION>/head.pt + metadata.json  (winning candidate, val metrics only)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.data.splits import load_manifest
from ml.embeddings.backbone import backbone_name, embedding_dim
from ml.embeddings.cache import load_embeddings
from ml.eval.harness import EmbeddedSpecimen
from ml.tracking.mlflow_utils import mlflow_run
from ml.train.isolation_guard import assert_no_test_ids
from ml.train.trainer import TrainConfig, train_head

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Versioned by backbone lineage: the prior "finetuned_arcface_v1" candidate
# was trained on 512-d OpenCLIP ViT-B-32 embeddings; this candidate is
# trained on 1024-d DINOv2 ViT-L/14 embeddings (see ml/embeddings/backbone.py)
# over the full gallery/val/test partitions + CV-preprocessed inputs, so it
# gets its own version id rather than overwriting an artifact with an
# incompatible input dimension. The prior candidate is preserved at
# ml/models/archive_openclip_v1/ for the promotion-gate "vs current active
# model" comparison.
CANDIDATE_VERSION = "finetuned_arcface_dinov2_v1"
EXPERIMENTS_DIR = "ml/eval/reports/experiments"
MODELS_DIR = "ml/models"
TRAIN_CACHE_DIR = "ml/data/embeddings_cache/train"


def _load_gallery_and_val(manifest: dict[str, Any]) -> tuple[list[EmbeddedSpecimen], list[EmbeddedSpecimen]]:
    vectors, metadata = load_embeddings()
    specimens = metadata["specimens"]
    gallery, val = [], []
    for specimen_id, vector in vectors.items():
        meta = specimens[specimen_id]
        item = EmbeddedSpecimen(id=specimen_id, label=meta["label"], label_name=meta["label_name"], vector=vector)
        if meta["split"] == "gallery":
            gallery.append(item)
        elif meta["split"] == "val":
            val.append(item)
    return gallery, val


def _load_train(manifest: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    vectors, metadata = load_embeddings(TRAIN_CACHE_DIR)
    specimens = metadata["specimens"]
    ids = sorted(vectors.keys())
    embeddings = np.stack([vectors[i] for i in ids], axis=0)
    labels = np.array([specimens[i]["label"] for i in ids], dtype=np.int64)
    base_ids = sorted({specimens[i]["base_id"] for i in ids})
    return embeddings, labels, base_ids


def _build_grid() -> list[TrainConfig]:
    """Small hyperparameter grid over lr / output dim / margin (ArcFace,
    the primary loss per PRD §18.3), plus one MLP-head variant and one
    triplet-loss comparison run (PRD §14.5 step 2)."""
    configs: list[TrainConfig] = []
    for lr in (1e-3, 3e-3):
        for output_dim in (128, 256):
            for margin in (0.3, 0.5):
                configs.append(
                    TrainConfig(
                        loss="arcface", lr=lr, head_hidden_dim=0, output_dim=output_dim,
                        margin=margin, scale=30.0, batch_size=128, max_epochs=40,
                        early_stop_patience=5, seed=42,
                    )
                )
    # Architecture comparison: one hidden layer instead of a pure linear head.
    configs.append(
        TrainConfig(
            loss="arcface", lr=1e-3, head_hidden_dim=256, output_dim=256,
            margin=0.3, scale=30.0, batch_size=128, max_epochs=40,
            early_stop_patience=5, seed=42,
        )
    )
    # Triplet comparison run (PRD §14.5 step 2: "start triplet/contrastive, compare against ArcFace").
    configs.append(
        TrainConfig(
            loss="triplet", lr=1e-3, head_hidden_dim=256, output_dim=256,
            margin=0.3, batch_size=128, max_epochs=40,
            early_stop_patience=5, seed=42,
        )
    )
    return configs


def _run_id(config: TrainConfig, index: int) -> str:
    return f"run{index:02d}_{config.loss}_lr{config.lr}_hd{config.head_hidden_dim}_od{config.output_dim}_m{config.margin}"


def main() -> None:
    manifest = load_manifest()
    dataset_hash = manifest["dataset_hash"]
    num_classes = manifest["num_classes"]
    input_dim = embedding_dim()

    gallery_specimens, val_specimens = _load_gallery_and_val(manifest)
    train_embeddings, train_labels, train_base_ids = _load_train(manifest)
    logger.info(
        "loaded gallery=%d val=%d train_views=%d (train_base_images=%d)",
        len(gallery_specimens), len(val_specimens), train_embeddings.shape[0], len(train_base_ids),
    )

    # Test-set isolation guard: every id this script reads must be train/val/gallery only.
    ids_used = (
        set(train_base_ids)
        | {s.id for s in val_specimens}
        | {s.id for s in gallery_specimens}
    )
    assert_no_test_ids(ids_used, manifest)
    logger.info("isolation guard passed: %d ids checked, none from test split", len(ids_used))

    Path(EXPERIMENTS_DIR).mkdir(parents=True, exist_ok=True)
    grid = _build_grid()
    logger.info("running %d-config sweep", len(grid))

    runs: list[dict[str, Any]] = []
    best_run: dict[str, Any] | None = None
    best_head_state = None

    with mlflow_run(
        f"finetune_sweep_{CANDIDATE_VERSION}",
        tags={"stage": "finetune_sweep", "candidate_version": CANDIDATE_VERSION},
        params={
            "backbone": backbone_name(),
            "dataset_hash": dataset_hash,
            "seed": manifest["seed"],
            "num_classes": num_classes,
            "input_dim": input_dim,
            "num_gallery": len(gallery_specimens),
            "num_val": len(val_specimens),
            "num_train_views": int(train_embeddings.shape[0]),
            "num_configs": len(grid),
        },
    ) as mlf:
        for i, config in enumerate(grid):
            run_id = _run_id(config, i)
            t0 = time.time()
            result = train_head(
                config, train_embeddings, train_labels, num_classes, gallery_specimens, val_specimens, input_dim
            )
            elapsed = time.time() - t0
            logger.info(
                "%s: best_epoch=%d val_recall@5=%.4f val_map@10=%.4f (%.1fs)",
                run_id, result.best_epoch, result.best_val_recall_at_5, result.final_val_metrics.get("map@10", 0.0), elapsed,
            )

            # Per-epoch val curve, namespaced by run_id so every hyperparameter
            # config's curve is visible (and comparable) in the same MLflow run.
            for point in result.epoch_curve:
                mlf.log_metric(f"{run_id}/train_loss", point["train_loss"], step=point["epoch"])
                mlf.log_metric(f"{run_id}/val_recall_at_5", point["val_recall@5"], step=point["epoch"])
            mlf.log_metric(f"{run_id}/best_val_recall_at_5", result.best_val_recall_at_5)
            mlf.log_metric(f"{run_id}/elapsed_seconds", elapsed)

            run_record = {
                "run_id": run_id,
                "config": asdict(config),
                "dataset_hash": dataset_hash,
                "seed": config.seed,
                "backbone": backbone_name(),
                "best_epoch": result.best_epoch,
                "best_val_recall_at_5": result.best_val_recall_at_5,
                "final_val_metrics": result.final_val_metrics,
                "epoch_curve": result.epoch_curve,
                "elapsed_seconds": elapsed,
                "train_ids": train_base_ids,
                "val_ids": [s.id for s in val_specimens],
                "gallery_ids": [s.id for s in gallery_specimens],
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            (Path(EXPERIMENTS_DIR) / f"{run_id}.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
            runs.append({k: v for k, v in run_record.items() if k not in ("epoch_curve", "train_ids", "val_ids", "gallery_ids")})

            if best_run is None or result.best_val_recall_at_5 > best_run["best_val_recall_at_5"] + 1e-9 or (
                abs(result.best_val_recall_at_5 - best_run["best_val_recall_at_5"]) <= 1e-9
                and result.final_val_metrics.get("map@10", 0.0) > best_run["final_val_metrics"].get("map@10", 0.0)
            ):
                best_run = run_record
                best_head_state = result.head_state_dict

        assert best_run is not None and best_head_state is not None

        runs_sorted = sorted(runs, key=lambda r: r["best_val_recall_at_5"], reverse=True)
        summary = {
            "dataset_hash": dataset_hash,
            "backbone": backbone_name(),
            "num_runs": len(runs),
            "winner_run_id": best_run["run_id"],
            "winner_val_metrics": best_run["final_val_metrics"],
            "runs_ranked": runs_sorted,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        summary_path = Path("ml/eval/reports/finetune_sweep_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("winner: %s val_recall@5=%.4f", best_run["run_id"], best_run["best_val_recall_at_5"])

        # Persist the winning candidate as a ModelVersion artifact — val metrics only (PRD §14.5 exit criteria).
        candidate_dir = Path(MODELS_DIR) / CANDIDATE_VERSION
        candidate_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_head_state, candidate_dir / "head.pt")
        candidate_metadata = {
            "model_version": CANDIDATE_VERSION,
            "base_model": backbone_name(),
            "method": best_run["config"]["loss"],
            "hyperparams": best_run["config"],
            "dataset_hash": dataset_hash,
            "seed": best_run["config"]["seed"],
            "input_dim": input_dim,
            "output_dim": best_run["config"]["output_dim"],
            "head_hidden_dim": best_run["config"]["head_hidden_dim"],
            "val_metrics": best_run["final_val_metrics"],
            "best_epoch": best_run["best_epoch"],
            "source_run_id": best_run["run_id"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": "val metrics only; test set not touched by this script (PRD 14.2 rule 3)",
        }
        metadata_path = candidate_dir / "metadata.json"
        metadata_path.write_text(json.dumps(candidate_metadata, indent=2), encoding="utf-8")
        logger.info("candidate ModelVersion saved to %s", candidate_dir)

        mlf.log_params(
            {
                "winner_run_id": best_run["run_id"],
                "winner_loss": best_run["config"]["loss"],
                "winner_output_dim": best_run["config"]["output_dim"],
            }
        )
        mlf.log_metrics({f"winner_val_{k}": v for k, v in best_run["final_val_metrics"].items() if v is not None})
        mlf.log_artifact(str(summary_path))
        mlf.log_artifact(str(metadata_path))
        mlf.log_artifact(str(candidate_dir / "head.pt"))


if __name__ == "__main__":
    main()
