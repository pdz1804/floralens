"""CLI entry point: Phase 3b — the ONE-SHOT held-out test evaluation of the
selected Phase 3 candidate, same protocol as the baseline (PRD §14.4, §14.8).

This is the only script in the whole pipeline that reads the `test` split.
It runs once, after model selection is already frozen on val
(`ml.scripts.run_finetune_sweep`), and never influences training or
hyperparameter choice — it only produces the number that decides promotion.

Usage:
    venv/Scripts/python.exe -m ml.scripts.run_test_eval
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from ml.data.splits import load_manifest
from ml.embeddings.backbone import backbone_name
from ml.embeddings.cache import load_embeddings
from ml.eval.harness import EmbeddedSpecimen, run_retrieval_eval
from ml.index.vector_store import VectorStore
from ml.train.model_io import load_candidate_head, project_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CANDIDATE_VERSION = "finetuned_arcface_v1"
MODELS_DIR = "ml/models"
REPORTS_DIR = "ml/eval/reports"


def main() -> None:
    manifest = load_manifest()
    candidate_dir = Path(MODELS_DIR) / CANDIDATE_VERSION
    head, candidate_metadata = load_candidate_head(candidate_dir)
    logger.info("loaded candidate %s (%s)", CANDIDATE_VERSION, candidate_metadata["method"])

    vectors, metadata = load_embeddings()
    specimens = metadata["specimens"]

    gallery_ids, gallery_raw, val_ids, val_raw, test_ids, test_raw = [], [], [], [], [], []
    gallery_meta, val_meta, test_meta = [], [], []
    for specimen_id, vector in vectors.items():
        meta = specimens[specimen_id]
        if meta["split"] == "gallery":
            gallery_ids.append(specimen_id)
            gallery_raw.append(vector)
            gallery_meta.append(meta)
        elif meta["split"] == "val":
            val_ids.append(specimen_id)
            val_raw.append(vector)
            val_meta.append(meta)
        elif meta["split"] == "test":
            test_ids.append(specimen_id)
            test_raw.append(vector)
            test_meta.append(meta)

    projected_gallery = project_embeddings(head, np.stack(gallery_raw, axis=0))
    projected_val = project_embeddings(head, np.stack(val_raw, axis=0))
    projected_test = project_embeddings(head, np.stack(test_raw, axis=0))

    def build_store(ids, projected, metas):
        store = VectorStore(model_version=CANDIDATE_VERSION)
        for id_, vec, meta in zip(ids, projected, metas):
            store.add(id_, vec, {"label": meta["label"], "label_name": meta["label_name"]})
        return store

    def build_queries(ids, projected, metas):
        return [
            EmbeddedSpecimen(id=id_, label=meta["label"], label_name=meta["label_name"], vector=vec)
            for id_, vec, meta in zip(ids, projected, metas)
        ]

    gallery_store = build_store(gallery_ids, projected_gallery, gallery_meta)

    report: dict = {
        "model_version": CANDIDATE_VERSION,
        "backbone": backbone_name(),
        "method": candidate_metadata["method"],
        "hyperparams": candidate_metadata["hyperparams"],
        "dataset_hash": manifest["dataset_hash"],
        "seed": candidate_metadata["seed"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gallery_size": len(gallery_store),
    }

    val_result = run_retrieval_eval(build_queries(val_ids, projected_val, val_meta), gallery_store)
    report["val"] = val_result["aggregate"]
    logger.info("candidate val (re-confirmed, same protocol): %s", report["val"])

    # The one-shot test read.
    test_result = run_retrieval_eval(build_queries(test_ids, projected_test, test_meta), gallery_store)
    report["test"] = test_result["aggregate"]
    logger.info("candidate TEST (one-shot): %s", report["test"])

    report["val_test_recall5_gap"] = abs(report["val"]["recall@5"] - report["test"]["recall@5"])
    report["overfitting_flag"] = report["val_test_recall5_gap"] > 0.05

    out_path = Path(REPORTS_DIR) / "candidate_test_eval_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("candidate test EvalReport saved to %s", out_path)


if __name__ == "__main__":
    main()
