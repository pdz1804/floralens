"""Unit tests for the promotion-gate decision logic (PRD §14.8)."""
from ml.train.promotion_gate import evaluate_promotion_gate


def test_candidate_worse_than_baseline_is_rejected():
    result = evaluate_promotion_gate(
        candidate_val_recall_at_5=0.80,
        candidate_test_recall_at_5=0.70,  # 10pt below baseline, well beyond tolerance
        baseline_test_recall_at_5=0.90,
        ece=0.03,
    )
    assert result.decision == "REJECT"
    assert any("below baseline" in r for r in result.reasons)


def test_candidate_better_than_baseline_is_promoted():
    result = evaluate_promotion_gate(
        candidate_val_recall_at_5=0.94,
        candidate_test_recall_at_5=0.93,
        baseline_test_recall_at_5=0.90,
        ece=0.02,
    )
    assert result.decision == "PROMOTE"


def test_candidate_within_tolerance_is_promoted():
    result = evaluate_promotion_gate(
        candidate_val_recall_at_5=0.905,
        candidate_test_recall_at_5=0.895,  # 0.5pt below baseline, within 2pt tolerance
        baseline_test_recall_at_5=0.90,
        ece=0.02,
    )
    assert result.decision == "PROMOTE"


def test_val_test_gap_blocks_promotion_even_if_recall_beats_baseline():
    result = evaluate_promotion_gate(
        candidate_val_recall_at_5=0.99,  # 10pt gap from test -> overfitting guard trips
        candidate_test_recall_at_5=0.92,
        baseline_test_recall_at_5=0.90,
        ece=0.02,
    )
    assert result.decision == "REJECT"
    assert any("overfitting" in r for r in result.reasons)


def test_high_ece_blocks_promotion():
    result = evaluate_promotion_gate(
        candidate_val_recall_at_5=0.93,
        candidate_test_recall_at_5=0.92,
        baseline_test_recall_at_5=0.90,
        ece=0.20,
    )
    assert result.decision == "REJECT"
    assert any("ECE" in r for r in result.reasons)


def test_result_serializes_to_dict():
    result = evaluate_promotion_gate(
        candidate_val_recall_at_5=0.93, candidate_test_recall_at_5=0.92, baseline_test_recall_at_5=0.90, ece=0.02
    )
    d = result.to_dict()
    assert d["decision"] == "PROMOTE"
    assert "thresholds" in d
