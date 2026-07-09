"""CLI entry point: Phase 3b — promotion gate (PRD §14.8). Compares the
candidate's one-shot test EvalReport against the active baseline's test
EvalReport, folds in the calibration ECE, and emits a PROMOTE/REJECT
decision with reasons. Does not re-read the test split — it only reads the
JSON reports already produced by `run_test_eval` and `run_calibration`.

Usage:
    venv/Scripts/python.exe -m ml.scripts.run_promotion_gate
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from ml.train.promotion_gate import evaluate_promotion_gate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("ml/eval/reports")
BASELINE_REPORT_PATH = REPORTS_DIR / "baseline_eval_report.json"
CANDIDATE_REPORT_PATH = REPORTS_DIR / "candidate_test_eval_report.json"
CALIBRATION_REPORT_PATH = REPORTS_DIR / "candidate_calibration_report.json"
DECISION_PATH = REPORTS_DIR / "promotion_decision.json"


def main() -> None:
    for path in (BASELINE_REPORT_PATH, CANDIDATE_REPORT_PATH, CALIBRATION_REPORT_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run run_baseline_eval / run_test_eval / run_calibration first"
            )

    baseline = json.loads(BASELINE_REPORT_PATH.read_text(encoding="utf-8"))
    candidate = json.loads(CANDIDATE_REPORT_PATH.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION_REPORT_PATH.read_text(encoding="utf-8"))

    result = evaluate_promotion_gate(
        candidate_val_recall_at_5=candidate["val"]["recall@5"],
        candidate_test_recall_at_5=candidate["test"]["recall@5"],
        baseline_test_recall_at_5=baseline["test"]["recall@5"],
        ece=calibration["ece_test_after_calibration"],
    )

    decision_report = {
        "decision": result.decision,
        "reasons": result.reasons,
        "comparison": {
            "baseline": {
                "model_version": baseline["model_version"],
                "test_recall@1": baseline["test"]["recall@1"],
                "test_recall@5": baseline["test"]["recall@5"],
                "test_recall@10": baseline["test"]["recall@10"],
                "test_map@10": baseline["test"]["map@10"],
                "test_mrr": baseline["test"]["mrr"],
            },
            "candidate": {
                "model_version": candidate["model_version"],
                "method": candidate["method"],
                "val_recall@5": candidate["val"]["recall@5"],
                "test_recall@1": candidate["test"]["recall@1"],
                "test_recall@5": candidate["test"]["recall@5"],
                "test_recall@10": candidate["test"]["recall@10"],
                "test_map@10": candidate["test"]["map@10"],
                "test_mrr": candidate["test"]["mrr"],
                "val_test_recall5_gap": result.val_test_recall5_gap,
            },
            "calibration_ece_test": calibration["ece_test_after_calibration"],
        },
        "thresholds": {
            "recall5_tolerance": result.recall5_tolerance,
            "val_test_gap_max": result.val_test_gap_max,
            "ece_max": result.ece_max,
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    DECISION_PATH.write_text(json.dumps(decision_report, indent=2), encoding="utf-8")

    logger.info("PROMOTION DECISION: %s", result.decision)
    for reason in result.reasons:
        logger.info("  - %s", reason)
    logger.info("saved to %s", DECISION_PATH)


if __name__ == "__main__":
    main()
