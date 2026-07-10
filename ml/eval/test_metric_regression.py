"""Build-blocking ML metric-regression gate (Gap G9).

FloraLens CI already guards leakage, test-set isolation, and calibration, but
nothing catches a *silent retrieval-quality regression* -- a new model (or a
re-run eval on the fixed subset) that quietly drops Recall@5 / mAP@10. This
test closes that gap: it fails the build if any guarded retrieval metric on the
FIXED test subset falls below its recorded baseline minus an allowed tolerance.

Design (offline / CPU-only, deterministic):
  * The reference values + tolerance live in a committed JSON threshold file
    (`ml/eval/metric_regression_baseline.json`), recorded from the currently
    shipped model's eval report.
  * The "current" metrics come from the committed eval report the threshold
    file points at (`ml/eval/reports/candidate_test_eval_report.json` -- the
    shipped finetuned_arcface_dinov2_v2 model). Those numbers were produced
    upstream by `ml.eval.harness` / `ml.eval.metrics`; the embeddings cache
    that would let us recompute them is gitignored and absent in CI, so the
    committed report IS the offline source of truth.
  * A real regression is caught because promoting a worse model regenerates
    that report with lower numbers, which then trip the assertion here.

This test reads only committed JSON with the standard library -- no torch, no
numpy, no model download, no GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# Repo root = three levels up from this file (ml/eval/test_metric_regression.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_PATH = _REPO_ROOT / "ml" / "eval" / "metric_regression_baseline.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_baseline() -> dict:
    assert _BASELINE_PATH.exists(), (
        f"metric-regression baseline file missing: {_BASELINE_PATH}. "
        "It is committed and required for the CI regression gate."
    )
    return _load_json(_BASELINE_PATH)


def test_baseline_threshold_file_is_well_formed():
    """The committed threshold file must carry everything the gate needs."""
    baseline = _load_baseline()
    for key in ("current_eval_report", "eval_split", "tolerance", "guarded_metrics"):
        assert key in baseline, f"baseline file is missing required key '{key}'"
    assert isinstance(baseline["tolerance"], (int, float)) and baseline["tolerance"] >= 0
    assert baseline["guarded_metrics"], "no guarded metrics declared -- the gate would be a no-op"


def test_current_eval_report_matches_baseline_dataset():
    """Guard against comparing metrics computed on a different eval subset:
    a silent dataset change invalidates any Recall@5 comparison."""
    baseline = _load_baseline()
    expected_hash = baseline.get("dataset_hash")
    if not expected_hash:
        pytest.skip("baseline records no dataset_hash to pin the eval subset")

    report_path = _REPO_ROOT / baseline["current_eval_report"]
    assert report_path.exists(), f"current eval report missing: {report_path}"
    report = _load_json(report_path)
    actual_hash = report.get("dataset_hash")
    assert actual_hash == expected_hash, (
        "current eval report was computed on a different dataset than the "
        f"recorded baseline (report dataset_hash={actual_hash!r}, "
        f"baseline dataset_hash={expected_hash!r}); Recall@5 comparison would "
        "be apples-to-oranges. Re-record the baseline for the new dataset."
    )


def test_retrieval_metrics_have_not_regressed():
    """FAIL LOUDLY if any guarded metric dropped below baseline - tolerance."""
    baseline = _load_baseline()
    split = baseline["eval_split"]
    tolerance = float(baseline["tolerance"])

    report_path = _REPO_ROOT / baseline["current_eval_report"]
    assert report_path.exists(), f"current eval report missing: {report_path}"
    report = _load_json(report_path)

    assert split in report, (
        f"eval report {report_path} has no '{split}' block; cannot read the "
        "metrics the gate guards."
    )
    current_metrics = report[split]

    regressions: list[str] = []
    for metric_name, reference_value in baseline["guarded_metrics"].items():
        assert metric_name in current_metrics, (
            f"guarded metric '{metric_name}' absent from the current eval "
            f"report's '{split}' block -- cannot verify no regression."
        )
        current_value = float(current_metrics[metric_name])
        floor = float(reference_value) - tolerance
        delta = current_value - float(reference_value)
        if current_value < floor:
            regressions.append(
                f"{metric_name}: current={current_value:.6f} "
                f"baseline={float(reference_value):.6f} "
                f"delta={delta:+.6f} floor(baseline-{tolerance:g})={floor:.6f}"
            )

    assert not regressions, (
        "Retrieval-quality REGRESSION on the fixed "
        f"{split} subset (model_version={report.get('model_version')!r}):\n  "
        + "\n  ".join(regressions)
        + "\nEither the change genuinely hurts retrieval quality (fix it), or "
        "a new model was intentionally promoted (re-record "
        "ml/eval/metric_regression_baseline.json from its eval report)."
    )
