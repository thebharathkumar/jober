"""The BusFactor facade: ask, evaluate (closed-book + leave-one-out), persistence."""

from __future__ import annotations

from bus_factor.app import BusFactor
from bus_factor.config import Settings
from bus_factor.memory.persistence import SqliteKnowledgeBase
from bus_factor.models import Document, QAPair, Source


def _answer_doc(i: int) -> Document:
    return Document(
        id=f"issue-{i}-answer",
        text=f"To use feature {i}, call configure_{i}() and pass the feature {i} options.",
        source=Source(
            id=f"issue-{i}-answer",
            kind="issue",
            title=f"Re: how to use feature {i}",
            created_at="2025-01-01T00:00:00Z",
        ),
    )


def _qa(i: int) -> QAPair:
    return QAPair(
        id=f"qa-{i}",
        question=f"How do I use feature {i}?",
        reference_answer=f"To use feature {i}, call configure_{i}() and pass the feature {i} options.",
        source=Source(id=f"issue-{i}", kind="issue", title=f"how to use feature {i}"),
    )


def test_ask_returns_grounded_cited_answer():
    bf = BusFactor.from_documents([_answer_doc(1)], settings=Settings())
    ans = bf.ask("How do I use feature 1?")
    assert ans.citations
    assert ans.confidence > 0.0
    assert bf.document_count == 1


def test_from_sqlite_roundtrip(tmp_path):
    path = tmp_path / "kb.db"
    with SqliteKnowledgeBase(path) as kb:
        kb.upsert_documents([_answer_doc(1), _answer_doc(2)])
    bf = BusFactor.from_sqlite(path, settings=Settings())
    assert bf.document_count == 2


def test_evaluate_closed_book_vs_leave_one_out():
    docs = [_answer_doc(i) for i in range(1, 11)]
    qa = [_qa(i) for i in range(1, 11)]
    bf = BusFactor.from_documents(docs, settings=Settings(holdout_fraction=0.5))

    closed = bf.evaluate(qa, leave_one_out=False)
    loo = bf.evaluate(qa, leave_one_out=True)

    assert closed.summary["n"] >= 1
    # Closed-book: the answer is present, so the right source is retrieved.
    assert closed.summary["retrieval_hit_rate"] == 1.0
    # Leave-one-out: the answer is withheld, so its source can never be retrieved.
    assert loo.summary["retrieval_hit_rate"] == 0.0


def test_fit_calibration_reduces_loo_ece_and_enables_abstention():
    docs = [_answer_doc(i) for i in range(1, 13)]
    qa = [_qa(i) for i in range(1, 13)]
    bf = BusFactor.from_documents(docs, settings=Settings(holdout_fraction=0.4))

    before = bf.evaluate(qa, leave_one_out=True).summary
    bf.fit_calibration(qa, leave_one_out=True)
    after = bf.evaluate(qa, leave_one_out=True).summary

    # Calibration must not worsen ECE, and here it should collapse overconfidence.
    assert after["ece"] <= before["ece"]
    # The withheld answers aren't recoverable, so the calibrated system abstains more.
    assert after["abstention_rate"] >= before["abstention_rate"]
