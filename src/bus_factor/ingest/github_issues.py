"""Turn a public repo's closed issues into knowledge + a ground-truth eval set.

The insight this module operationalizes: a maintainer's closed issues are a
*pre-built evaluation set*. Someone asks a question; the maintainer answers;
the issue closes. That is a real question paired with a real authoritative
answer — thousands of them, for free.

We identify the authoritative answer using GitHub's ``author_association`` on
each comment (``OWNER`` / ``MEMBER`` / ``COLLABORATOR``), which is a far more
reliable signal than string-matching usernames.

I/O and transforms are deliberately split:

* ``fetch_*`` functions do network calls (retrying, rate-limit aware).
* ``issue_to_document`` / ``build_qa_pair`` are pure and fully unit-tested.

That boundary is what lets CI verify the extraction logic with fixtures and no
network, which is the whole reason the eval numbers can be trusted.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

from bus_factor.errors import IngestError, RateLimitError, RepoNotFoundError
from bus_factor.logging_config import get_logger
from bus_factor.models import Document, QAPair, Source

log = get_logger(__name__)

AUTHORITATIVE = {"OWNER", "MEMBER", "COLLABORATOR"}
_MIN_ANSWER_CHARS = 80
_MIN_QUESTION_CHARS = 20
_API = "https://api.github.com"
_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_MAX_RETRIES = 4
_TIMEOUT_S = 30


def validate_repo(repo: str) -> str:
    """Validate an ``owner/name`` slug before it goes anywhere near a URL.

    Guards against malformed input and path-injection into the API URL.
    """
    if not _REPO_RE.match(repo):
        raise IngestError(f"invalid repo slug {repo!r}; expected 'owner/name'")
    return repo


# --- pure transforms (unit-tested) ------------------------------------------


def _clean(text: str | None) -> str:
    return (text or "").strip()


def looks_like_question(title: str, body: str) -> bool:
    """Cheap heuristic: is this issue actually a question worth grading on?"""
    blob = f"{title} {body}".lower()
    if "?" in blob:
        return True
    starters = ("how ", "why ", "what ", "when ", "can i", "is it", "should i", "does ")
    return title.lower().startswith(starters)


def issue_to_document(issue: dict[str, Any]) -> Document:
    """Represent the issue itself (title + body) as a retrievable document."""
    number = issue.get("number", 0)
    src = Source(
        id=f"issue-{number}",
        kind="issue",
        title=_clean(issue.get("title")),
        url=_clean(issue.get("html_url")),
        author=_clean((issue.get("user") or {}).get("login")),
        created_at=_clean(issue.get("created_at")),
    )
    return Document(id=f"issue-{number}", text=_clean(issue.get("body")), source=src)


def authoritative_answer(comments: list[dict[str, Any]]) -> dict[str, Any] | None:
    """First substantive comment written by someone with authority over the repo."""
    for c in comments:
        is_authority = c.get("author_association") in AUTHORITATIVE
        if is_authority and len(_clean(c.get("body"))) >= _MIN_ANSWER_CHARS:
            return c
    return None


def answer_to_document(issue: dict[str, Any], comment: dict[str, Any]) -> Document:
    """The maintainer's answer, as its own high-value knowledge document."""
    number = issue.get("number", 0)
    src = Source(
        id=f"issue-{number}-answer",
        kind="issue",
        title=f"Re: {_clean(issue.get('title'))}",
        url=_clean(comment.get("html_url")) or _clean(issue.get("html_url")),
        author=_clean((comment.get("user") or {}).get("login")),
        created_at=_clean(comment.get("created_at")),
    )
    return Document(id=src.id, text=_clean(comment.get("body")), source=src)


def build_qa_pair(issue: dict[str, Any], comments: list[dict[str, Any]]) -> QAPair | None:
    """Produce a ground-truth QA pair, or ``None`` if the issue doesn't qualify.

    Qualifies when: the issue reads like a question, and a repo authority left a
    substantive answer. The answer author is excluded from being the same as the
    asker to avoid self-answered issues polluting the set.
    """
    title = _clean(issue.get("title"))
    body = _clean(issue.get("body"))
    if len(f"{title} {body}") < _MIN_QUESTION_CHARS or not looks_like_question(title, body):
        return None

    answer = authoritative_answer(comments)
    if answer is None:
        return None

    asker = _clean((issue.get("user") or {}).get("login"))
    answerer = _clean((answer.get("user") or {}).get("login"))
    if asker and asker == answerer:
        return None

    number = issue.get("number", 0)
    src = Source(
        id=f"issue-{number}",
        kind="issue",
        title=title,
        url=_clean(issue.get("html_url")),
        author=answerer,
        created_at=_clean(issue.get("created_at")),
    )
    question = title if not body else f"{title}\n\n{body}"
    return QAPair(
        id=f"qa-issue-{number}",
        question=question,
        reference_answer=_clean(answer.get("body")),
        source=src,
        tags=[lbl.get("name", "") for lbl in issue.get("labels", []) if isinstance(lbl, dict)],
    )


def transform_repo(
    issues_with_comments: list[tuple[dict[str, Any], list[dict[str, Any]]]],
) -> tuple[list[Document], list[QAPair]]:
    """Pure end-to-end transform: (issue, comments) pairs -> knowledge docs + eval pairs.

    Modeling decision: the knowledge base is what the *subject authored* — their
    answers — not the questions other people asked them. Indexing the asker's
    issue body as "knowledge" both pollutes the corpus with non-subject text and
    makes an extractive answerer echo the question back. So only the authoritative
    answer becomes a Document; the issue itself lives on solely as an eval question.
    """
    docs: list[Document] = []
    qa: list[QAPair] = []
    for issue, comments in issues_with_comments:
        if issue.get("pull_request"):
            continue  # the issues endpoint also returns PRs; skip them
        ans = authoritative_answer(comments)
        if ans is not None:
            docs.append(answer_to_document(issue, ans))
        pair = build_qa_pair(issue, comments)
        if pair is not None:
            qa.append(pair)
    return docs, qa


# --- network I/O (best-effort) ----------------------------------------------


def _handle_http_error(exc: urllib.error.HTTPError, url: str) -> float | None:
    """Map an HTTPError to either a raised typed error or a retry-delay.

    Returns the number of seconds to wait before retrying, or raises a terminal
    error the caller should not retry.
    """
    if exc.code == 404:
        raise RepoNotFoundError(f"not found or inaccessible: {url}") from exc
    if exc.code in (403, 429):
        remaining = exc.headers.get("X-RateLimit-Remaining")
        if remaining == "0":
            reset = exc.headers.get("X-RateLimit-Reset")
            reset_epoch = int(reset) if reset and reset.isdigit() else None
            raise RateLimitError(
                "GitHub API rate limit exhausted; set GITHUB_TOKEN or wait for reset",
                reset_epoch=reset_epoch,
            ) from exc
        retry_after = exc.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            return float(retry_after)  # secondary/abuse rate limit: honor and retry
        # A 403 with quota remaining and no Retry-After is an authorization or
        # egress-policy denial, not a transient failure. Retrying it wastes time
        # and (per the egress-proxy contract) is exactly what not to do.
        raise IngestError(
            f"forbidden (403) for {url}: the authenticated identity is not authorized "
            "for this resource, or an egress policy blocks it — not retrying"
        ) from exc
    if 500 <= exc.code < 600:
        return None  # transient server error -> exponential backoff
    raise IngestError(f"GitHub API error {exc.code} for {url}") from exc


def _get(url: str, token: str = "", *, retries: int = _MAX_RETRIES) -> Any:
    """GET JSON with exponential backoff and rate-limit awareness."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url)  # noqa: S310 (fixed https host)
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("User-Agent", "bus-factor-ingest")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_exc = exc
            delay = _handle_http_error(exc, url)  # raises on terminal errors
            wait = delay if delay is not None else 2.0**attempt
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_exc = exc
            wait = 2.0**attempt
        if attempt < retries - 1:
            log.warning("request failed (%s), retrying in %.1fs", last_exc, wait)
            time.sleep(wait)
    raise IngestError(f"exhausted {retries} retries for {url}: {last_exc}")


def fetch_closed_issues(
    repo: str, token: str = "", max_issues: int = 100, per_page: int = 50
) -> list[dict[str, Any]]:
    """Fetch closed issues (newest first). ``repo`` is ``owner/name``."""
    validate_repo(repo)
    issues: list[dict[str, Any]] = []
    page = 1
    while len(issues) < max_issues:
        url = (
            f"{_API}/repos/{repo}/issues"
            f"?state=closed&per_page={per_page}&page={page}&sort=updated&direction=desc"
        )
        batch = _get(url, token)
        if not batch:
            break
        issues.extend(b for b in batch if not b.get("pull_request"))
        log.info("fetched %d issues so far from %s", len(issues), repo)
        page += 1
        time.sleep(0.2)  # be polite to the rate limiter
    return issues[:max_issues]


def fetch_comments(repo: str, number: int, token: str = "") -> list[dict[str, Any]]:
    validate_repo(repo)
    return _get(f"{_API}/repos/{repo}/issues/{number}/comments?per_page=100", token)


def ingest_repo(
    repo: str, token: str = "", max_issues: int = 100
) -> tuple[list[Document], list[QAPair]]:
    """Orchestrate fetch + transform for a public repo. Network-dependent."""
    validate_repo(repo)
    issues = fetch_closed_issues(repo, token, max_issues=max_issues)
    paired: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for issue in issues:
        try:
            comments = fetch_comments(repo, issue["number"], token)
        except (IngestError, KeyError) as exc:
            log.warning("skipping comments for issue %s: %s", issue.get("number"), exc)
            comments = []
        paired.append((issue, comments))
        time.sleep(0.1)
    log.info("ingested %d issues from %s", len(paired), repo)
    return transform_repo(paired)
