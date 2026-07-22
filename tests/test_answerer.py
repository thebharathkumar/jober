"""End-to-end answerer behaviour on the offline provider."""

from __future__ import annotations

from datetime import date

from bus_factor.agent.answerer import Answerer
from bus_factor.config import Settings
from bus_factor.memory.store import MemoryStore
from bus_factor.models import Document, Source


def _store() -> MemoryStore:
    store = MemoryStore()
    store.add(
        [
            Document(
                id="issue-1-answer",
                text=(
                    "Set the DATAKIT_CACHE environment variable to any path and datakit "
                    "will use it for all cached artifacts. It takes effect at import time."
                ),
                source=Source(
                    id="issue-1-answer",
                    kind="issue",
                    title="Re: How do I change the cache directory?",
                    url="https://example.com/1",
                    created_at="2024-01-01T00:00:00Z",
                ),
            ),
            Document(
                id="issue-2-answer",
                text="Use datakit.stream() to process large files in constant memory.",
                source=Source(
                    id="issue-2-answer",
                    kind="issue",
                    title="Re: streaming large files",
                    created_at="2026-06-01T00:00:00Z",
                ),
            ),
        ]
    )
    return store


def test_answer_is_grounded_and_cited():
    answerer = Answerer(_store(), Settings())
    ans = answerer.answer("How do I change the cache directory?", as_of=date(2026, 7, 22))
    assert "DATAKIT_CACHE" in ans.text
    assert ans.citations
    assert ans.citations[0].document_id == "issue-1-answer"
    assert ans.confidence > 0.0


def test_staleness_uses_freshest_evidence_and_flags_old():
    answerer = Answerer(_store(), Settings(staleness_warn_days=365))
    ans = answerer.answer("How do I change the cache directory?", as_of=date(2026, 7, 22))
    # Freshest cited evidence is the 2024 doc (~2.5y) -> should flag stale.
    assert ans.staleness_days is not None
    assert ans.meta["stale"] is True


def test_no_evidence_returns_zero_confidence_abstention():
    answerer = Answerer(MemoryStore(), Settings())
    ans = answerer.answer("Anything at all?")
    assert ans.confidence == 0.0
    assert ans.citations == []
    assert ans.meta["n_evidence"] == 0
