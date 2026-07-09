"""CLI entry point: Phase 3b — promotion gate (PRD §14.8), extended to gate
against **both** the zero-shot baseline AND the currently active model
(task requirement: "Promotion gate vs the current active model + baseline").
Compares the candidate's one-shot test EvalReport against each, folds in the
calibration ECE, and emits a PROMOTE/REJECT decision with reasons. Does not
re-read the test split — it only reads the JSON reports already produced by
`run_baseline_eval`, `run_test_eval`, and `run_calibration`. Also writes the
model card (JSON + Markdown) summarizing data/backbone/metrics/calibration/
decision for this ModelVersion, and logs the decision to MLflow.

Usage:
    venv/Scripts/python.exe -m ml.scripts.run_promotion_gate
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ml.tracking.mlflow_utils import mlflow_run
from ml.train.promotion_gate import PromotionGateResult, evaluate_promotion_gate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("ml/eval/reports")
MODELS_DIR = Path("ml/models")
MANIFEST_PATH = Path("ml/data/manifests/split_manifest.json")
EMBEDDINGS_METADATA_PATH = Path("ml/data/embeddings_cache/metadata.json")

BASELINE_REPORT_PATH = REPORTS_DIR / "baseline_eval_report.json"
CANDIDATE_REPORT_PATH = REPORTS_DIR / "candidate_test_eval_report.json"
CALIBRATION_REPORT_PATH = REPORTS_DIR / "candidate_calibration_report.json"
DECISION_PATH = REPORTS_DIR / "promotion_decision.json"

# The previously active model (OpenCLIP-backbone candidate), archived before
# this pipeline run overwrote ml/models/finetuned_arcface_v1 with a new,
# dimensionally-incompatible candidate (see run_finetune_sweep.py docstring).
# If absent (e.g. a from-scratch run with no prior active model), the gate
# falls back to comparing against the zero-shot baseline only.
ACTIVE_MODEL_REPORT_PATH = REPORTS_DIR / "archive" / "openclip_v1" / "candidate_test_eval_report.json"


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _combine(
    vs_baseline: PromotionGateResult, vs_active: PromotionGateResult | None
) -> tuple[str, list[str]]:
    """PROMOTE only if the candidate clears every gate that was evaluated."""
    results = [vs_baseline] + ([vs_active] if vs_active is not None else [])
    reasons: list[str] = []
    for label, result in zip(["baseline"] + (["active model"] if vs_active else []), results):
        reasons.extend(f"[vs {label}] {r}" for r in result.reasons)
    decision = "PROMOTE" if all(r.decision == "PROMOTE" for r in results) else "REJECT"
    return decision, reasons


def _write_model_card(
    candidate: dict, calibration: dict, decision_report: dict, candidate_metadata: dict
) -> None:
    manifest = _load(MANIFEST_PATH) or {}
    embeddings_meta = _load(EMBEDDINGS_METADATA_PATH) or {}
    split_counts: dict[str, int] = {}
    for meta in embeddings_meta.get("specimens", {}).values():
        split_counts[meta["split"]] = split_counts.get(meta["split"], 0) + 1

    card = {
        "model_version": candidate["model_version"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data": {
            "dataset": manifest.get("dataset", "oxford-102-flowers"),
            "dataset_hash": manifest.get("dataset_hash"),
            "num_classes": manifest.get("num_classes"),
            "embedded_splits": split_counts,
            "total_embedded": sum(split_counts.values()),
        },
        "backbone": {
            "name": candidate["backbone"],
            "frozen": True,
            "input_dim": candidate_metadata.get("input_dim"),
        },
        "method": candidate["method"],
        "hyperparams": candidate["hyperparams"],
        "seed": candidate["seed"],
        "metrics": {"val": candidate["val"], "test": candidate["test"]},
        "val_test_recall5_gap": candidate["val_test_recall5_gap"],
        "calibration": {
            "method": calibration.get("calibration_method"),
            "ece_test_before": calibration.get("ece_test_before_calibration"),
            "ece_test_after": calibration.get("ece_test_after_calibration"),
            "band_thresholds": calibration.get("band_thresholds"),
        },
        "promotion": decision_report,
    }
    card_dir = MODELS_DIR / candidate["model_version"]
    card_dir.mkdir(parents=True, exist_ok=True)
    (card_dir / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    md_lines = [
        f"# Model Card — {card['model_version']}",
        "",
        f"Generated: {card['created_at']}",
        "",
        "## Data",
        f"- Dataset: {card['data']['dataset']} (hash `{card['data']['dataset_hash']}`)",
        f"- Classes: {card['data']['num_classes']}",
        f"- Embedded splits: {card['data']['embedded_splits']} (total {card['data']['total_embedded']})",
        "",
        "## Backbone",
        f"- {card['backbone']['name']} (frozen, input dim {card['backbone']['input_dim']})",
        "",
        "## Training",
        f"- Method: {card['method']}",
        f"- Hyperparameters: `{json.dumps(card['hyperparams'])}`",
        f"- Seed: {card['seed']}",
        "",
        "## Evaluation (retrieval protocol, PRD §14.4)",
        f"- Val: {json.dumps(card['metrics']['val'])}",
        f"- Test (one-shot): {json.dumps(card['metrics']['test'])}",
        f"- val<->test Recall@5 gap: {card['val_test_recall5_gap']:.4f}",
        "",
        "## Calibration",
        f"- Method: {card['calibration']['method']}",
        f"- ECE (test) before: {card['calibration']['ece_test_before']:.4f}",
        f"- ECE (test) after: {card['calibration']['ece_test_after']:.4f}",
        "",
        "## Promotion decision",
        f"- Decision: **{card['promotion']['decision']}**",
        *[f"- {reason}" for reason in card["promotion"]["reasons"]],
        "",
    ]
    (card_dir / "model_card.md").write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("model card written to %s/model_card.{json,md}", card_dir)


def main() -> None:
    for path in (BASELINE_REPORT_PATH, CANDIDATE_REPORT_PATH, CALIBRATION_REPORT_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run run_baseline_eval / run_test_eval / run_calibration first"
            )

    baseline = _load(BASELINE_REPORT_PATH)
    candidate = _load(CANDIDATE_REPORT_PATH)
    calibration = _load(CALIBRATION_REPORT_PATH)
    active_model = _load(ACTIVE_MODEL_REPORT_PATH)
    candidate_metadata = _load(MODELS_DIR / candidate["model_version"] / "metadata.json") or {}

    ece = calibration["ece_test_after_calibration"]

    vs_baseline = evaluate_promotion_gate(
        candidate_val_recall_at_5=candidate["val"]["recall@5"],
        candidate_test_recall_at_5=candidate["test"]["recall@5"],
        baseline_test_recall_at_5=baseline["test"]["recall@5"],
        ece=ece,
    )
    vs_active = None
    if active_model is not None:
        vs_active = evaluate_promotion_gate(
            candidate_val_recall_at_5=candidate["val"]["recall@5"],
            candidate_test_recall_at_5=candidate["test"]["recall@5"],
            baseline_test_recall_at_5=active_model["test"]["recall@5"],
            ece=ece,
        )

    decision, combined_reasons = _combine(vs_baseline, vs_active)

    decision_report = {
        "decision": decision,
        "reasons": combined_reasons,
        "comparison": {
            "baseline": {
                "model_version": baseline["model_version"],
                "test_recall@1": baseline["test"]["recall@1"],
                "test_recall@5": baseline["test"]["recall@5"],
                "test_recall@10": baseline["test"]["recall@10"],
                "test_map@10": baseline["test"]["map@10"],
                "test_mrr": baseline["test"]["mrr"],
            },
            "active_model": (
                {
                    "model_version": active_model["model_version"],
                    "test_recall@5": active_model["test"]["recall@5"],
                }
                if active_model is not None
                else None
            ),
            "candidate": {
                "model_version": candidate["model_version"],
                "method": candidate["method"],
                "val_recall@5": candidate["val"]["recall@5"],
                "test_recall@1": candidate["test"]["recall@1"],
                "test_recall@5": candidate["test"]["recall@5"],
                "test_recall@10": candidate["test"]["recall@10"],
                "test_map@10": candidate["test"]["map@10"],
                "test_mrr": candidate["test"]["mrr"],
                "val_test_recall5_gap": vs_baseline.val_test_recall5_gap,
            },
            "calibration_ece_test": ece,
        },
        "thresholds": {
            "recall5_tolerance": vs_baseline.recall5_tolerance,
            "val_test_gap_max": vs_baseline.val_test_gap_max,
            "ece_max": vs_baseline.ece_max,
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    DECISION_PATH.write_text(json.dumps(decision_report, indent=2), encoding="utf-8")

    _write_model_card(candidate, calibration, decision_report, candidate_metadata)

    with mlflow_run(
        f"promotion_gate_{candidate['model_version']}",
        tags={"stage": "promotion_gate", "candidate_version": candidate["model_version"]},
        params={
            "candidate_version": candidate["model_version"],
            "baseline_version": baseline["model_version"],
            "active_model_version": active_model["model_version"] if active_model else None,
        },
    ) as mlf:
        mlf.log_metric("candidate_test_recall_at_5", candidate["test"]["recall@5"])
        mlf.log_metric("baseline_test_recall_at_5", baseline["test"]["recall@5"])
        if active_model is not None:
            mlf.log_metric("active_model_test_recall_at_5", active_model["test"]["recall@5"])
        mlf.log_metric("calibration_ece_test", ece)
        mlf.log_metric("val_test_recall5_gap", vs_baseline.val_test_recall5_gap)
        mlf.set_tag("promotion_decision", decision)
        mlf.log_artifact(str(DECISION_PATH))
        mlf.log_artifact(str(MODELS_DIR / candidate["model_version"] / "model_card.json"))
        mlf.log_artifact(str(MODELS_DIR / candidate["model_version"] / "model_card.md"))

    logger.info("PROMOTION DECISION: %s", decision)
    for reason in combined_reasons:
        logger.info("  - %s", reason)
    logger.info("saved to %s", DECISION_PATH)


if __name__ == "__main__":
    main()
