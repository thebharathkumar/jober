"""Turn a public repo's closed issues into knowledge + a ground-truth eval set.

The insight this module operationalizes: a maintainer's closed issues are a
*pre-built evaluation set*. Someone asks a question; the maintainer answers;
the issue closes. That is a real question paired with a real authoritative
answer — thousands of them, for free.

We identify the authoritative answer using GitHub's ``author_association`` on
each comment (``OWNER`` / ``MEMBER`` / ``COLLABORATOR``), which is a far more
reliable signal than string-matching usernames.

I/O and transforms are deliberately split:

* ``fetch_*`` functions do network calls (untested, best-effort, retry-light).
* ``issue_to_document`` / ``build_qa_pair`` are pure and fully unit-tested.

That boundary is what lets CI verify the extraction logic with fixtures and no
network, which is the whole reason the eval numbers can be trusted.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from bus_factor.models import Document, QAPair, Source

AUTHORITATIVE = {"OWNER", "MEMBER", "COLLABORATOR"}
_MIN_ANSWER_CHARS = 80
_MIN_QUESTION_CHARS = 20
_API = "https://api.github.com"


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


def _get(url: str, token: str = "") -> Any:
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "bus-factor-ingest")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (trusted host)
        return json.loads(resp.read().decode("utf-8"))


def fetch_closed_issues(
    repo: str, token: str = "", max_issues: int = 100, per_page: int = 50
) -> list[dict[str, Any]]:
    """Fetch closed issues (newest first). ``repo`` is ``owner/name``."""
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
        page += 1
        time.sleep(0.2)  # be polite to the rate limiter
    return issues[:max_issues]


def fetch_comments(repo: str, number: int, token: str = "") -> list[dict[str, Any]]:
    return _get(f"{_API}/repos/{repo}/issues/{number}/comments?per_page=100", token)


def ingest_repo(
    repo: str, token: str = "", max_issues: int = 100
) -> tuple[list[Document], list[QAPair]]:
    """Orchestrate fetch + transform for a public repo. Network-dependent."""
    issues = fetch_closed_issues(repo, token, max_issues=max_issues)
    paired: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for issue in issues:
        try:
            comments = fetch_comments(repo, issue["number"], token)
        except (urllib.error.URLError, KeyError):
            comments = []
        paired.append((issue, comments))
        time.sleep(0.1)
    return transform_repo(paired)
