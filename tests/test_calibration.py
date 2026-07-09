"""Unit tests for score calibration: monotonicity + ECE improvement vs raw score."""
import numpy as np

from ml.eval.calibration import confidence_band, expected_calibration_error, fit_calibrator


def _synthetic_pairs(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Raw cosine scores that are rank-correlated with truth but a poor
    probability estimate on their own: true P(match) = sigmoid(8*score - 4),
    while the raw score itself (e.g. clipped to [0,1]) systematically
    over/under-states that probability, so an honest calibrator should
    reduce ECE relative to using the raw score directly as a confidence.
    """
    rng = np.random.default_rng(seed)
    scores = rng.uniform(-1.0, 1.0, size=n)
    true_prob = 1.0 / (1.0 + np.exp(-(8 * scores - 4)))
    labels = (rng.uniform(0, 1, size=n) < true_prob).astype(np.int64)
    return scores, labels


def test_isotonic_calibrator_is_monotonic_in_raw_score():
    scores, labels = _synthetic_pairs()
    calibrator = fit_calibrator(scores, labels, method="isotonic")

    probe = np.linspace(-1.0, 1.0, 50)
    predicted = calibrator.predict(probe)
    # Monotonic non-decreasing: no adjacent pair should decrease.
    assert np.all(np.diff(predicted) >= -1e-9)


def test_platt_calibrator_is_monotonic_in_raw_score():
    scores, labels = _synthetic_pairs()
    calibrator = fit_calibrator(scores, labels, method="platt")

    probe = np.linspace(-1.0, 1.0, 50)
    predicted = calibrator.predict(probe)
    assert np.all(np.diff(predicted) >= -1e-9)


def test_calibration_reduces_ece_vs_uncalibrated_raw_score():
    train_scores, train_labels = _synthetic_pairs(seed=1)
    test_scores, test_labels = _synthetic_pairs(seed=2)

    calibrator = fit_calibrator(train_scores, train_labels, method="isotonic")
    calibrated_confidence = calibrator.predict(test_scores)
    ece_after, _ = expected_calibration_error(calibrated_confidence, test_labels)

    # Naive uncalibrated "confidence": raw cosine rescaled from [-1,1] to [0,1].
    uncalibrated_confidence = (test_scores + 1.0) / 2.0
    ece_before, _ = expected_calibration_error(uncalibrated_confidence, test_labels)

    assert ece_after < ece_before


def test_expected_calibration_error_perfect_calibration_is_zero():
    # Confidence exactly matches empirical frequency within each bin.
    confidences = np.array([0.05] * 100 + [0.95] * 100)
    labels = np.array([0] * 95 + [1] * 5 + [1] * 95 + [0] * 5)
    ece, bins = expected_calibration_error(confidences, labels, n_bins=10)
    assert ece < 0.02
    assert len(bins) == 10


def test_fit_calibrator_requires_both_classes():
    import pytest

    with pytest.raises(ValueError):
        fit_calibrator(np.array([0.1, 0.2, 0.3]), np.array([1, 1, 1]))


def test_confidence_band_thresholds():
    assert confidence_band(0.9) == "high"
    assert confidence_band(0.70) == "high"
    assert confidence_band(0.5) == "medium"
    assert confidence_band(0.40) == "medium"
    assert confidence_band(0.1) == "low"
