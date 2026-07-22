"""Tests for the pure ingest transforms (no network)."""

from __future__ import annotations

from bus_factor.ingest.github_issues import (
    authoritative_answer,
    build_qa_pair,
    issue_to_document,
    looks_like_question,
    transform_repo,
)


def _issue(number=1, title="How do I do X?", body="Some detail?", author="asker"):
    return {
        "number": number,
        "title": title,
        "body": body,
        "html_url": f"https://github.com/acme/repo/issues/{number}",
        "user": {"login": author},
        "created_at": "2025-01-01T00:00:00Z",
        "labels": [{"name": "question"}],
    }


def _comment(body, association="OWNER", author="maintainer"):
    return {
        "body": body,
        "author_association": association,
        "user": {"login": author},
        "html_url": "https://github.com/acme/repo/issues/1#issuecomment-1",
        "created_at": "2025-01-02T00:00:00Z",
    }


def test_looks_like_question_detects_question_mark_and_starters():
    assert looks_like_question("Does it work?", "")
    assert looks_like_question("How do I configure it", "no q mark here")
    assert not looks_like_question("Crash on startup", "stack trace")


def test_authoritative_answer_prefers_repo_authority_and_length():
    short = _comment("thanks!", association="OWNER")
    community = _comment("A" * 200, association="NONE", author="rando")
    good = _comment("B" * 200, association="MEMBER", author="dev")
    assert authoritative_answer([short, community, good]) is good


def test_build_qa_pair_happy_path():
    answer_body = "You configure it with an environment variable, set before import." * 2
    qa = build_qa_pair(_issue(), [_comment(answer_body)])
    assert qa is not None
    assert qa.reference_answer.startswith("You configure it")
    assert qa.source.url.endswith("/issues/1")
    assert "question" in qa.tags


def test_build_qa_pair_rejects_self_answered():
    body = "Here is the answer to my own question, which is long enough to qualify." * 2
    qa = build_qa_pair(
        _issue(author="same"),
        [_comment(body, association="OWNER", author="same")],
    )
    assert qa is None


def test_build_qa_pair_rejects_non_question():
    body = "This is a substantive maintainer reply that is definitely long enough here." * 2
    qa = build_qa_pair(
        _issue(title="Segfault in loader", body="no question at all"),
        [_comment(body)],
    )
    assert qa is None


def test_issue_to_document_carries_provenance():
    doc = issue_to_document(_issue(number=7))
    assert doc.id == "issue-7"
    assert doc.source.url.endswith("/issues/7")
    assert doc.source.kind == "issue"


def test_transform_repo_skips_pull_requests():
    pr = _issue(number=9)
    pr["pull_request"] = {"url": "..."}
    long_answer = "A genuinely substantive answer body that clears the length gate." * 2
    docs, qa = transform_repo([(pr, [_comment(long_answer)])])
    assert docs == []
    assert qa == []
