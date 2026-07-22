"""Configuration validation fails fast on bad input."""

from __future__ import annotations

import pytest

from bus_factor.config import Settings
from bus_factor.errors import ConfigError


def test_defaults_are_valid():
    s = Settings()
    assert s.top_k >= 1
    assert 0.0 < s.holdout_fraction < 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 0},
        {"rrf_k": 0},
        {"max_tokens": 0},
        {"holdout_fraction": 0.0},
        {"holdout_fraction": 1.0},
        {"judge_f1_threshold": 1.5},
        {"low_confidence_threshold": -0.1},
        {"staleness_warn_days": -1},
    ],
)
def test_invalid_settings_raise_config_error(kwargs):
    with pytest.raises(ConfigError):
        Settings(**kwargs)
