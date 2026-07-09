"""Score calibration (PRD §14.6): map raw cosine similarity -> P(same species).

Fit on **val** pairs only, evaluated on **test** pairs (never fit on test).
Raw cosine is not a probability — nothing enforces that a 0.9 cosine match is
"90% likely correct". A calibrator (isotonic regression, the default, or
Platt/logistic scaling) learns that mapping from (score, is_same_class)
pairs, and Expected Calibration Error (ECE) measures how well the mapped
value tracks empirical accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ml.eval.harness import EmbeddedSpecimen
from ml.index.vector_store import VectorStore

CalibrationMethod = Literal["isotonic", "platt"]

# Confidence banding thresholds on the *calibrated* probability (PRD §14.6,
# §A3: banding is derived from calibrated probability, not raw cosine).
# Chosen as round, interpretable cut points; not tuned on test data.
BAND_HIGH_THRESHOLD = 0.70
BAND_MEDIUM_THRESHOLD = 0.40


class Calibrator(Protocol):
    def predict(self, scores: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class FittedCalibrator:
    method: CalibrationMethod
    _model: Any

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        if self.method == "isotonic":
            return np.clip(self._model.predict(scores.ravel()), 0.0, 1.0)
        return np.clip(self._model.predict_proba(scores)[:, 1], 0.0, 1.0)


def build_query_gallery_pairs(
    queries: list[EmbeddedSpecimen], gallery_store: VectorStore
) -> tuple[np.ndarray, np.ndarray]:
    """All (query, gallery item) cosine scores + same-class labels.

    Using the full cross product (not just top-K) avoids biasing the
    calibrator toward only high-confidence matches — it needs to see the
    full range of scores, including confident-but-wrong and unconfident-
    but-right pairs, to learn an honest mapping.
    """
    scores: list[float] = []
    labels: list[int] = []
    for q in queries:
        results = gallery_store.query(q.vector, top_k=len(gallery_store), exclude_ids={q.id})
        for r in results:
            scores.append(r.score)
            labels.append(1 if r.metadata["label"] == q.label else 0)
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def fit_calibrator(scores: np.ndarray, labels: np.ndarray, method: CalibrationMethod = "isotonic") -> FittedCalibrator:
    """Fit a calibrator on (raw_score, is_same_class) pairs. Callers must pass
    only val-split pairs (PRD §14.2 rule 5: calibrator fit on val, never test).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.shape[0] != labels.shape[0]:
        raise ValueError("scores and labels must be the same length")
    if scores.shape[0] == 0:
        raise ValueError("cannot fit a calibrator on zero pairs")
    if len(set(labels.tolist())) < 2:
        raise ValueError("calibration requires both positive and negative pairs")

    if method == "isotonic":
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(scores, labels)
    elif method == "platt":
        model = LogisticRegression()
        model.fit(scores.reshape(-1, 1), labels)
    else:
        raise ValueError(f"unknown calibration method: {method}")
    return FittedCalibrator(method=method, _model=model)


def expected_calibration_error(
    confidences: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> tuple[float, list[dict[str, float]]]:
    """ECE + per-bin reliability data.

    ECE = sum_bins (|bin| / N) * |mean_confidence(bin) - empirical_accuracy(bin)|.
    Bins are fixed-width over [0, 1] (standard ECE definition).
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if confidences.shape[0] != labels.shape[0]:
        raise ValueError("confidences and labels must be the same length")
    n = confidences.shape[0]
    if n == 0:
        return 0.0, []

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)
        count = int(in_bin.sum())
        if count == 0:
            bins.append(
                {"bin_lower": float(lo), "bin_upper": float(hi), "mean_confidence": 0.0, "empirical_accuracy": 0.0, "count": 0}
            )
            continue
        mean_conf = float(confidences[in_bin].mean())
        acc = float(labels[in_bin].mean())
        ece += (count / n) * abs(mean_conf - acc)
        bins.append(
            {"bin_lower": float(lo), "bin_upper": float(hi), "mean_confidence": mean_conf, "empirical_accuracy": acc, "count": count}
        )
    return float(ece), bins


def confidence_band(probability: float) -> str:
    """Map a calibrated probability to high/medium/low banding (PRD A3)."""
    if probability >= BAND_HIGH_THRESHOLD:
        return "high"
    if probability >= BAND_MEDIUM_THRESHOLD:
        return "medium"
    return "low"
