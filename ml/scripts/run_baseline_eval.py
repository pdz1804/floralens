"""CLI entry point: run the retrieval eval protocol (PRD §14.4) for the
frozen zero-shot backbone on val and test, against the gallery index, and
save a combined baseline EvalReport JSON.

Usage:
    venv/Scripts/python.exe -m ml.scripts.run_baseline_eval
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_VERSION = "baseline"
REPORTS_DIR = "ml/eval/reports"


def _build_gallery_store(vectors: dict[str, np.ndarray], metadata: dict) -> VectorStore:
    store = VectorStore(model_version=MODEL_VERSION)
    specimens = metadata["specimens"]
    for specimen_id, vector in vectors.items():
        meta = specimens[specimen_id]
        if meta["split"] != "gallery":
            continue
        store.add(specimen_id, vector, {"label": meta["label"], "label_name": meta["label_name"]})
    return store


def _build_queries(
    vectors: dict[str, np.ndarray], metadata: dict, split: str
) -> list[EmbeddedSpecimen]:
    specimens = metadata["specimens"]
    queries = []
    for specimen_id, vector in vectors.items():
        meta = specimens[specimen_id]
        if meta["split"] != split:
            continue
        queries.append(
            EmbeddedSpecimen(
                id=specimen_id, label=meta["label"], label_name=meta["label_name"], vector=vector
            )
        )
    return queries


def main() -> None:
    manifest = load_manifest()
    vectors, metadata = load_embeddings()
    gallery_store = _build_gallery_store(vectors, metadata)
    logger.info("gallery indexed: %d specimens", len(gallery_store))

    report: dict = {
        "model_version": MODEL_VERSION,
        "backbone": backbone_name(),
        "dataset_hash": manifest["dataset_hash"],
        "seed": manifest["seed"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gallery_size": len(gallery_store),
    }

    for split in ("val", "test"):
        queries = _build_queries(vectors, metadata, split)
        logger.info("evaluating split=%s with %d queries", split, len(queries))
        result = run_retrieval_eval(queries, gallery_store)
        report[split] = result["aggregate"]
        report[f"{split}_per_query_count"] = len(result["per_query"])
        logger.info("%s aggregate: %s", split, result["aggregate"])

    # Anti-metric guardrail check (PRD §5): val<->test Recall@5 gap.
    val_r5 = report["val"]["recall@5"]
    test_r5 = report["test"]["recall@5"]
    report["val_test_recall5_gap"] = abs(val_r5 - test_r5)
    report["overfitting_flag"] = report["val_test_recall5_gap"] > 0.05

    out_dir = Path(REPORTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_eval_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("baseline EvalReport saved to %s", out_path)
    logger.info(
        "val Recall@1=%.3f Recall@5=%.3f mAP@10=%.3f MRR=%.3f",
        report["val"]["recall@1"],
        report["val"]["recall@5"],
        report["val"]["map@10"],
        report["val"]["mrr"],
    )
    logger.info(
        "test Recall@1=%.3f Recall@5=%.3f mAP@10=%.3f MRR=%.3f",
        report["test"]["recall@1"],
        report["test"]["recall@5"],
        report["test"]["map@10"],
        report["test"]["mrr"],
    )


if __name__ == "__main__":
    main()
