"""Runtime configuration, read once from the environment.

Keeping this in one place (rather than reaching for ``os.environ`` all over the
codebase) means the whole system's behaviour is described by a single object
that is trivial to print, log, or override in tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from bus_factor.errors import ConfigError


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass
class Settings:
    # --- retrieval ---
    top_k: int = 5
    # Reciprocal-rank-fusion constant for hybrid retrieval.
    rrf_k: int = 60

    # --- provenance / trust thresholds ---
    # Answers whose freshest evidence is older than this are flagged stale.
    staleness_warn_days: int = 365
    # Below this retrieval-derived confidence we recommend routing to a human.
    low_confidence_threshold: float = 0.35

    # --- llm provider ---
    # If unset, the pipeline uses the offline extractive provider so evals and
    # demos run with zero credentials.
    anthropic_api_key: str = ""
    model: str = "claude-sonnet-5"
    max_tokens: int = 700

    # --- eval ---
    # Fraction of QA pairs held out for evaluation (deterministic hash split).
    holdout_fraction: float = 0.3
    # Token-F1 above which the offline lexical judge calls an answer correct.
    judge_f1_threshold: float = 0.4

    # --- github ingest ---
    github_token: str = ""

    def __post_init__(self) -> None:
        """Fail fast on nonsensical configuration rather than deep in a run."""
        if self.top_k < 1:
            raise ConfigError(f"top_k must be >= 1, got {self.top_k}")
        if self.rrf_k < 1:
            raise ConfigError(f"rrf_k must be >= 1, got {self.rrf_k}")
        if self.max_tokens < 1:
            raise ConfigError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if not 0.0 < self.holdout_fraction < 1.0:
            raise ConfigError(
                f"holdout_fraction must be in (0, 1), got {self.holdout_fraction}"
            )
        if not 0.0 <= self.judge_f1_threshold <= 1.0:
            raise ConfigError(
                f"judge_f1_threshold must be in [0, 1], got {self.judge_f1_threshold}"
            )
        if not 0.0 <= self.low_confidence_threshold <= 1.0:
            raise ConfigError(
                f"low_confidence_threshold must be in [0, 1], got {self.low_confidence_threshold}"
            )
        if self.staleness_warn_days < 0:
            raise ConfigError(
                f"staleness_warn_days must be >= 0, got {self.staleness_warn_days}"
            )

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            top_k=_int("BF_TOP_K", 5),
            rrf_k=_int("BF_RRF_K", 60),
            staleness_warn_days=_int("BF_STALENESS_WARN_DAYS", 365),
            low_confidence_threshold=_float("BF_LOW_CONFIDENCE", 0.35),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            model=os.environ.get("BF_MODEL", "claude-sonnet-5"),
            max_tokens=_int("BF_MAX_TOKENS", 700),
            holdout_fraction=_float("BF_HOLDOUT_FRACTION", 0.3),
            judge_f1_threshold=_float("BF_JUDGE_F1", 0.4),
            github_token=os.environ.get("GITHUB_TOKEN", ""),
        )

    @property
    def has_llm(self) -> bool:
        return bool(self.anthropic_api_key)
