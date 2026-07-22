"""Hardened GitHub client: repo validation and HTTP error mapping (no network)."""

from __future__ import annotations

import email.message
import urllib.error

import pytest

from bus_factor.errors import IngestError, RateLimitError, RepoNotFoundError
from bus_factor.ingest.github_issues import _handle_http_error, validate_repo


def _http_error(code: int, headers: dict[str, str] | None = None) -> urllib.error.HTTPError:
    msg = email.message.Message()
    for k, v in (headers or {}).items():
        msg[k] = v
    return urllib.error.HTTPError(url="https://api.github.com/x", code=code, msg="err", hdrs=msg, fp=None)


def test_validate_repo_accepts_valid_slug():
    assert validate_repo("simonw/datasette") == "simonw/datasette"


@pytest.mark.parametrize("bad", ["not-a-repo", "a/b/c", "owner/", "/name", "owner name"])
def test_validate_repo_rejects_bad_slug(bad):
    with pytest.raises(IngestError):
        validate_repo(bad)


def test_404_maps_to_repo_not_found():
    with pytest.raises(RepoNotFoundError):
        _handle_http_error(_http_error(404), "url")


def test_exhausted_rate_limit_raises_with_reset():
    hdrs = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1700000000"}
    with pytest.raises(RateLimitError) as excinfo:
        _handle_http_error(_http_error(403, hdrs), "url")
    assert excinfo.value.reset_epoch == 1700000000


def test_server_error_signals_retry():
    # 5xx returns a delay (None -> caller uses exponential backoff), not an error.
    assert _handle_http_error(_http_error(503), "url") is None


def test_throttle_with_retry_after_returns_delay():
    hdrs = {"Retry-After": "7"}  # 403 but not rate-limit-exhausted
    assert _handle_http_error(_http_error(403, hdrs), "url") == 7.0


def test_other_4xx_raises_ingest_error():
    with pytest.raises(IngestError):
        _handle_http_error(_http_error(422), "url")
