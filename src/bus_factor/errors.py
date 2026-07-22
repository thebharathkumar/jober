"""Typed exception hierarchy.

A single root (``BusFactorError``) means callers can catch everything this
library raises with one ``except`` and, crucially, distinguish *our* failures
from unexpected bugs. Each subtype maps to a distinct, actionable failure so the
CLI can exit with a clear message instead of a stack trace.
"""

from __future__ import annotations


class BusFactorError(Exception):
    """Base class for every error this library raises deliberately."""


class ConfigError(BusFactorError):
    """Invalid configuration (bad env var, out-of-range setting)."""


class DataError(BusFactorError):
    """Malformed or unreadable input data (bad JSONL row, missing field)."""


class IngestError(BusFactorError):
    """A source could not be ingested (network, auth, not found)."""


class RepoNotFoundError(IngestError):
    """The requested repository does not exist or is not accessible."""


class RateLimitError(IngestError):
    """The upstream API rate limit was hit.

    ``reset_epoch`` is the UNIX time the limit resets, when the API provides it.
    """

    def __init__(self, message: str, reset_epoch: int | None = None) -> None:
        super().__init__(message)
        self.reset_epoch = reset_epoch


class ProviderError(BusFactorError):
    """The answer/judge provider failed in a way we cannot recover from."""
