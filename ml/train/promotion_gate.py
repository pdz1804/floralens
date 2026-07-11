"""Promotion gate (PRD §14.8): decide whether a candidate model version
replaces the currently active one, using only held-out **test** metrics
(computed exactly once, upstream of this module) plus the val/test gap and
calibration quality.

Gate criteria (all must hold to PROMOTE):
  1. candidate test Recall@5 >= baseline test Recall@5 - `recall5_tolerance`.
     PRD §14.8 rule 1 asks for strictly no regression; this module accepts a
     small, explicitly-declared tolerance (default 2 points) rather than a
     hard >= so a statistically-noisy, near-identical candidate is not
     rejected on a fractional-point difference computed over a few hundred
     test queries. The tolerance is a parameter, not a hidden fudge — it is
     logged in every `PromotionGateResult`.
  2. |val Recall@5 - test Recall@5| <= `val_test_gap_max` (overfitting guard,
     PRD §5 anti-metric / §14.8 rule 2).
  3. calibration ECE <= `ece_max` (PRD §14.8 rule 3).
Any failing criterion rejects the candidate; the previous active version
(baseline, or a prior promoted version) stays active.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Decision = Literal["PROMOTE", "REJECT"]

DEFAULT_RECALL5_TOLERANCE = 0.02
DEFAULT_VAL_TEST_GAP_MAX = 0.05
DEFAULT_ECE_MAX = 0.05


@dataclass(frozen=True)
class PromotionGateResult:
    """Outcome of `evaluate_promotion_gate`: the PROMOTE/REJECT decision plus
    every metric and threshold that fed it, so the decision is fully
    auditable from the result alone.
    """

    decision: Decision
    reasons: list[str]
    candidate_val_recall_at_5: float
    candidate_test_recall_at_5: float
    baseline_test_recall_at_5: float
    recall5_delta: float
    val_test_recall5_gap: float
    ece: float
    recall5_tolerance: float
    val_test_gap_max: float
    ece_max: float

    def to_dict(self) -> dict:
        """Render as a JSON-serializable dict (thresholds nested together) for
        writing the gate decision to a report file."""
        return {
            "decision": self.decision,
            "reasons": self.reasons,
            "candidate_val_recall_at_5": self.candidate_val_recall_at_5,
            "candidate_test_recall_at_5": self.candidate_test_recall_at_5,
            "baseline_test_recall_at_5": self.baseline_test_recall_at_5,
            "recall5_delta": self.recall5_delta,
            "val_test_recall5_gap": self.val_test_recall5_gap,
            "ece": self.ece,
            "thresholds": {
                "recall5_tolerance": self.recall5_tolerance,
                "val_test_gap_max": self.val_test_gap_max,
                "ece_max": self.ece_max,
            },
        }


def evaluate_promotion_gate(
    candidate_val_recall_at_5: float,
    candidate_test_recall_at_5: float,
    baseline_test_recall_at_5: float,
    ece: float,
    recall5_tolerance: float = DEFAULT_RECALL5_TOLERANCE,
    val_test_gap_max: float = DEFAULT_VAL_TEST_GAP_MAX,
    ece_max: float = DEFAULT_ECE_MAX,
) -> PromotionGateResult:
    """Decide PROMOTE vs REJECT for a candidate against the three gate criteria
    (see module docstring): bounded Recall@5 regression, val/test gap, ECE.

    All three criteria must pass to PROMOTE; any single failure REJECTs, and
    every triggered reason is collected (not just the first) so a rejected
    candidate's report explains all of its shortfalls at once.
    """
    val_test_gap = abs(candidate_val_recall_at_5 - candidate_test_recall_at_5)
    recall5_delta = candidate_test_recall_at_5 - baseline_test_recall_at_5

    reasons: list[str] = []
    if recall5_delta < -recall5_tolerance:
        reasons.append(
            f"candidate test Recall@5 ({candidate_test_recall_at_5:.4f}) is more than "
            f"{recall5_tolerance:.4f} below baseline ({baseline_test_recall_at_5:.4f})"
        )
    if val_test_gap > val_test_gap_max:
        reasons.append(
            f"val<->test Recall@5 gap ({val_test_gap:.4f}) exceeds tolerance ({val_test_gap_max:.4f}); "
            "likely overfitting to val"
        )
    if ece > ece_max:
        reasons.append(f"calibration ECE ({ece:.4f}) exceeds max ({ece_max:.4f})")

    decision: Decision = "REJECT" if reasons else "PROMOTE"
    if not reasons:
        reasons = [
            f"candidate test Recall@5 ({candidate_test_recall_at_5:.4f}) within tolerance of baseline "
            f"({baseline_test_recall_at_5:.4f}); val/test gap and ECE within bounds"
        ]

    return PromotionGateResult(
        decision=decision,
        reasons=reasons,
        candidate_val_recall_at_5=candidate_val_recall_at_5,
        candidate_test_recall_at_5=candidate_test_recall_at_5,
        baseline_test_recall_at_5=baseline_test_recall_at_5,
        recall5_delta=recall5_delta,
        val_test_recall5_gap=val_test_gap,
        ece=ece,
        recall5_tolerance=recall5_tolerance,
        val_test_gap_max=val_test_gap_max,
        ece_max=ece_max,
    )
