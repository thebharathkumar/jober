"""Metric primitives and calibration."""

from __future__ import annotations

from bus_factor.eval.metrics import (
    expected_calibration_error,
    normalized_exact_match,
    rouge_l,
    token_f1,
)


def test_token_f1_identical_is_one():
    assert token_f1("set the cache dir", "set the cache dir") == 1.0


def test_token_f1_disjoint_is_zero():
    assert token_f1("apples oranges", "quantum tunnelling") == 0.0


def test_token_f1_partial_overlap_between_zero_and_one():
    score = token_f1("set the cache directory", "set the cache path")
    assert 0.0 < score < 1.0


def test_rouge_l_rewards_in_order_overlap():
    ordered = rouge_l("a b c d", "a b c d")
    shuffled = rouge_l("d c b a", "a b c d")
    assert ordered == 1.0
    assert shuffled < ordered


def test_normalized_exact_match_ignores_case_and_spacing():
    assert normalized_exact_match("Set  the CACHE", "set the cache") == 1.0


def test_ece_perfectly_calibrated_is_low():
    # Confidence equals accuracy in each region -> ECE ~ 0.
    confidences = [0.9, 0.9, 0.9, 0.9, 0.1, 0.1]
    correct = [True, True, True, True, False, False]
    assert expected_calibration_error(confidences, correct, n_bins=10) < 0.15


def test_ece_miscalibrated_is_high():
    # High confidence but always wrong -> large gap.
    confidences = [0.95, 0.95, 0.95, 0.95]
    correct = [False, False, False, False]
    assert expected_calibration_error(confidences, correct, n_bins=10) > 0.8
