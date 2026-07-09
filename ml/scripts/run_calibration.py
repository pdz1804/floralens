"""CLI entry point: Phase 3b — fit the score calibrator on VAL, evaluate ECE
on TEST (PRD §14.6). Reads the same candidate produced by
`run_finetune_sweep` and projects gallery/val/test embeddings through it,
exactly like `run_test_eval`, so calibration is scored against the same
candidate whose retrieval metrics are reported.

Usage:
    venv/Scripts/python.exe -m ml.scripts.run_calibration
"""
from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np

from ml.eval.calibration import build_query_gallery_pairs, confidence_band, expected_calibration_error, fit_calibrator
from ml.eval.harness import EmbeddedSpecimen
from ml.embeddings.cache import load_embeddings
from ml.index.vector_store import VectorStore
from ml.train.model_io import load_candidate_head, project_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CANDIDATE_VERSION = "finetuned_arcface_v1"
MODELS_DIR = "ml/models"
REPORTS_DIR = "ml/eval/reports"
CALIBRATION_METHOD = "isotonic"


def _split_vectors(vectors: dict, metadata: dict) -> dict[str, dict]:
    specimens = metadata["specimens"]
    out: dict[str, dict] = {"gallery": {}, "val": {}, "test": {}}
    for specimen_id, vector in vectors.items():
        meta = specimens[specimen_id]
        split = meta["split"]
        if split in out:
            out[split][specimen_id] = (vector, meta)
    return out


def _build_store_and_queries(head, split_data: dict[str, tuple]) -> tuple[VectorStore, list[EmbeddedSpecimen]]:
    ids = list(split_data.keys())
    raw = np.stack([split_data[i][0] for i in ids], axis=0)
    projected = project_embeddings(head, raw)
    store = VectorStore(model_version=CANDIDATE_VERSION)
    queries = []
    for id_, vec in zip(ids, projected):
        meta = split_data[id_][1]
        store.add(id_, vec, {"label": meta["label"], "label_name": meta["label_name"]})
        queries.append(EmbeddedSpecimen(id=id_, label=meta["label"], label_name=meta["label_name"], vector=vec))
    return store, queries


def main() -> None:
    candidate_dir = Path(MODELS_DIR) / CANDIDATE_VERSION
    head, candidate_metadata = load_candidate_head(candidate_dir)

    vectors, metadata = load_embeddings()
    by_split = _split_vectors(vectors, metadata)

    # A single gallery store (candidate-projected gallery vectors); val and
    # test queries are each scored against this same index (PRD §14.4).
    gallery_store, _ = _build_store_and_queries(head, by_split["gallery"])
    _, val_queries = _build_store_and_queries(head, by_split["val"])
    _, test_queries = _build_store_and_queries(head, by_split["test"])

    val_scores, val_labels = build_query_gallery_pairs(val_queries, gallery_store)
    logger.info("val calibration pairs: %d (positive rate=%.4f)", len(val_scores), val_labels.mean())

    calibrator = fit_calibrator(val_scores, val_labels, method=CALIBRATION_METHOD)
    # Persist the fitted calibrator next to the head weights so
    # apps/api/app/search_service can load it at query time if/when this
    # candidate is promoted (MODEL_VERSION env switch). Written regardless
    # of the eventual promotion decision — harmless if never activated.
    (candidate_dir / "calibrator.pkl").write_bytes(pickle.dumps(calibrator))

    # Uncalibrated baseline: naive rescale of raw cosine [-1,1] -> [0,1].
    uncalibrated_val_conf = (val_scores + 1.0) / 2.0
    ece_val_before, _ = expected_calibration_error(uncalibrated_val_conf, val_labels)
    calibrated_val_conf = calibrator.predict(val_scores)
    ece_val_after, val_bins = expected_calibration_error(calibrated_val_conf, val_labels)

    test_scores, test_labels = build_query_gallery_pairs(test_queries, gallery_store)
    logger.info("test calibration pairs: %d (positive rate=%.4f)", len(test_scores), test_labels.mean())
    uncalibrated_test_conf = (test_scores + 1.0) / 2.0
    ece_test_before, _ = expected_calibration_error(uncalibrated_test_conf, test_labels)
    calibrated_test_conf = calibrator.predict(test_scores)
    ece_test_after, test_bins = expected_calibration_error(calibrated_test_conf, test_labels)

    band_counts = {"high": 0, "medium": 0, "low": 0}
    for p in calibrated_test_conf:
        band_counts[confidence_band(float(p))] += 1

    report = {
        "model_version": CANDIDATE_VERSION,
        "calibration_method": CALIBRATION_METHOD,
        "fit_on_split": "val",
        "evaluated_on_split": "test",
        "num_val_pairs": int(len(val_scores)),
        "num_test_pairs": int(len(test_scores)),
        "ece_val_before_calibration": ece_val_before,
        "ece_val_after_calibration": ece_val_after,
        "ece_test_before_calibration": ece_test_before,
        "ece_test_after_calibration": ece_test_after,
        "test_reliability_bins": test_bins,
        "band_thresholds": {"high": 0.70, "medium": 0.40},
        "test_band_counts": band_counts,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = Path(REPORTS_DIR) / "candidate_calibration_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "calibration: ECE test before=%.4f after=%.4f (target <= 0.05); saved to %s",
        ece_test_before, ece_test_after, out_path,
    )


if __name__ == "__main__":
    main()
