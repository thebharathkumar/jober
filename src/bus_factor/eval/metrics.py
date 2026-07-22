"""Metric primitives. Pure functions, stdlib only, individually unit-tested.

We report several complementary numbers rather than one, because a single
headline metric is easy to game and easy to misread:

* token-F1 / ROUGE-L  -> content overlap with the reference answer
* exact-ish match      -> strict agreement (rare, but honest to report)
* ECE                  -> is the system's confidence *calibrated*? This is the
                          metric that matters for a trust layer: high confidence
                          must actually mean more-often-right.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token F1 between prediction and reference."""
    pred = tokenize(prediction)
    ref = tokenize(reference)
    if not pred or not ref:
        return float(pred == ref)  # both empty -> 1.0, else 0.0

    common: dict[str, int] = {}
    ref_counts: dict[str, int] = {}
    for t in ref:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    overlap = 0
    for t in pred:
        if ref_counts.get(t, 0) - common.get(t, 0) > 0:
            common[t] = common.get(t, 0) + 1
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 based on longest common subsequence of tokens."""
    pred = tokenize(prediction)
    ref = tokenize(reference)
    if not pred or not ref:
        return float(pred == ref)
    # LCS length via DP.
    dp = [[0] * (len(ref) + 1) for _ in range(len(pred) + 1)]
    for i in range(1, len(pred) + 1):
        for j in range(1, len(ref) + 1):
            if pred[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[len(pred)][len(ref)]
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def normalized_exact_match(prediction: str, reference: str) -> float:
    def norm(s: str) -> str:
        return " ".join(tokenize(s))

    return float(norm(prediction) == norm(reference))


def expected_calibration_error(
    confidences: list[float], correct: list[bool], n_bins: int = 10
) -> float:
    """ECE: weighted gap between confidence and accuracy across probability bins.

    0 is perfectly calibrated. This is the number that tells you whether the
    confidence score is meaningful or decorative.
    """
    if not confidences:
        return 0.0
    n = len(confidences)
    ece = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        idx = [
            i
            for i, c in enumerate(confidences)
            if (c > lo or (b == 0 and c >= lo)) and c <= hi
        ]
        if not idx:
            continue
        avg_conf = sum(confidences[i] for i in idx) / len(idx)
        acc = sum(1 for i in idx if correct[i]) / len(idx)
        ece += (len(idx) / n) * abs(avg_conf - acc)
    return ece


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
