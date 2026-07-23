"""Confidence calibration via isotonic regression.

The leave-one-out eval shows the raw retrieval-derived confidence is overconfident
under distribution shift: high scores that are often wrong. Calibration fixes the
*numbers* so a reported confidence of 0.7 means "right about 70% of the time."

We fit a monotonic (non-decreasing) map from raw score -> empirical correctness on
a held-out calibration split, using the Pool Adjacent Violators Algorithm (PAVA).
Isotonic regression is the right tool here: it assumes only that higher retrieval
confidence should not mean *lower* accuracy (monotonicity), and otherwise lets the
data dictate the shape — no sigmoid assumption, no parameters to tune.

Implemented from scratch to keep the core dependency-free.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field


def _pava(values: list[float], weights: list[float]) -> list[float]:
    """Weighted Pool Adjacent Violators: least-squares isotonic (non-decreasing) fit.

    Returns a non-decreasing sequence aligned to the input order, each entry the
    weighted mean of the pooled block it belongs to. Callers aggregate by unique
    x *before* calling this, so ties in x are handled by weight, not by ordering.
    """
    # each block: [value, total_weight, member_count]
    blocks: list[list[float]] = []
    for val, wt in zip(values, weights, strict=True):
        cur = [val, wt, 1.0]
        while blocks and blocks[-1][0] > cur[0]:
            prev = blocks.pop()
            total = prev[1] + cur[1]
            cur = [(prev[0] * prev[1] + cur[0] * cur[1]) / total, total, prev[2] + cur[2]]
        blocks.append(cur)
    out: list[float] = []
    for value, _wt, count in blocks:
        out.extend([value] * int(count))
    return out


@dataclass
class IsotonicCalibrator:
    """Maps a raw confidence in [0, 1] to a calibrated probability in [0, 1]."""

    x: list[float] = field(default_factory=list)  # sorted raw scores
    y: list[float] = field(default_factory=list)  # isotonic-fitted probabilities
    fitted: bool = False

    def fit(self, raw: list[float], correct: list[bool]) -> IsotonicCalibrator:
        if not raw:
            return self
        # Aggregate outcomes by unique raw score (weighted by count) before the
        # isotonic fit — the textbook way to handle tied x values. Without this,
        # many points sharing one confidence get spuriously split into a 0->1 ramp.
        agg: dict[float, list[float]] = {}
        for score, c in zip(raw, correct, strict=True):
            bucket = agg.setdefault(score, [0.0, 0.0])  # [sum_correct, count]
            bucket[0] += 1.0 if c else 0.0
            bucket[1] += 1.0
        xs = sorted(agg)
        means = [agg[x][0] / agg[x][1] for x in xs]
        weights = [agg[x][1] for x in xs]
        self.x = xs
        self.y = _pava(means, weights)
        self.fitted = True
        return self

    def predict(self, value: float) -> float:
        """Interpolate the calibrated probability for a raw score."""
        if not self.fitted or not self.x:
            return value  # identity until fitted
        if value <= self.x[0]:
            return self.y[0]
        if value >= self.x[-1]:
            return self.y[-1]
        i = bisect.bisect_right(self.x, value)
        x0, x1 = self.x[i - 1], self.x[i]
        y0, y1 = self.y[i - 1], self.y[i]
        if x1 == x0:
            return y1
        return y0 + (y1 - y0) * (value - x0) / (x1 - x0)

    def to_dict(self) -> dict[str, object]:
        return {"x": self.x, "y": self.y, "fitted": self.fitted}

    @classmethod
    def from_dict(cls, d: dict) -> IsotonicCalibrator:
        return cls(x=list(d.get("x", [])), y=list(d.get("y", [])), fitted=bool(d.get("fitted")))
