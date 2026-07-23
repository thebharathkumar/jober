"""Isotonic calibrator: monotonicity, separability, and ECE reduction."""

from __future__ import annotations

from bus_factor.calibration import IsotonicCalibrator, _pava
from bus_factor.eval.metrics import expected_calibration_error


def test_pava_produces_non_decreasing_output():
    out = _pava([0.0, 1.0, 0.0, 0.0, 1.0, 1.0], [1.0] * 6)
    assert out == sorted(out)  # isotonic == non-decreasing


def test_unfitted_calibrator_is_identity():
    cal = IsotonicCalibrator()
    assert cal.predict(0.42) == 0.42


def test_separable_signal_maps_low_to_zero_high_to_one():
    # Raw < 0.5 always wrong, raw > 0.5 always right -> clean step.
    raw = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    correct = [False, False, False, False, True, True, True, True]
    cal = IsotonicCalibrator().fit(raw, correct)
    assert cal.predict(0.2) < 0.2
    assert cal.predict(0.85) > 0.8


def test_prediction_is_monotonic_in_raw_score():
    raw = [0.1, 0.3, 0.5, 0.7, 0.9]
    correct = [False, False, True, True, True]
    cal = IsotonicCalibrator().fit(raw, correct)
    preds = [cal.predict(v) for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
    assert preds == sorted(preds)


def test_calibration_reduces_ece_on_overconfident_data():
    # Overconfident: high raw scores, but only right ~half the time.
    raw = [0.9] * 10 + [0.8] * 10
    correct = [True, False] * 5 + [True, False] * 5  # 50% correct in each bucket
    ece_before = expected_calibration_error(raw, correct)

    cal = IsotonicCalibrator().fit(raw, correct)
    calibrated = [cal.predict(v) for v in raw]
    ece_after = expected_calibration_error(calibrated, correct)

    assert ece_after < ece_before
    assert ece_after < 0.15  # calibrated confidence ~0.5 matches ~50% accuracy


def test_roundtrips_through_dict():
    cal = IsotonicCalibrator().fit([0.2, 0.8], [False, True])
    restored = IsotonicCalibrator.from_dict(cal.to_dict())
    assert restored.fitted
    assert restored.predict(0.8) == cal.predict(0.8)
