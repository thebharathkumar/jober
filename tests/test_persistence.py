"""SQLite knowledge base: roundtrip, incremental upsert, and stats."""

from __future__ import annotations

from bus_factor.memory.persistence import SqliteKnowledgeBase
from bus_factor.models import Document, QAPair, Source


def _doc(i: int, text: str = "hello world") -> Document:
    return Document(
        id=f"d{i}",
        text=text,
        source=Source(id=f"s{i}", kind="issue", title=f"t{i}", created_at="2025-01-01T00:00:00Z"),
    )


def _qa(i: int) -> QAPair:
    return QAPair(
        id=f"q{i}",
        question="how?",
        reference_answer="like this",
        source=Source(id=f"s{i}", kind="issue", title=f"t{i}"),
    )


def test_roundtrip_documents_and_qa(tmp_path):
    with SqliteKnowledgeBase(tmp_path / "kb.db") as kb:
        kb.upsert_documents([_doc(1), _doc(2)])
        kb.upsert_qa_pairs([_qa(1)])
        docs = kb.documents()
        pairs = kb.qa_pairs()
    assert {d.id for d in docs} == {"d1", "d2"}
    assert docs[0].source.created_at == "2025-01-01T00:00:00Z"
    assert [p.id for p in pairs] == ["q1"]


def test_incremental_upsert_updates_in_place(tmp_path):
    path = tmp_path / "kb.db"
    with SqliteKnowledgeBase(path) as kb:
        kb.upsert_documents([_doc(1, text="v1")])
    # Re-open (simulating a second ingest run) and upsert the same id.
    with SqliteKnowledgeBase(path) as kb:
        kb.upsert_documents([_doc(1, text="v2")])
        docs = kb.documents()
    assert len(docs) == 1
    assert docs[0].text == "v2"


def test_stats_counts_by_kind(tmp_path):
    with SqliteKnowledgeBase(tmp_path / "kb.db") as kb:
        kb.upsert_documents([_doc(1), _doc(2)])
        kb.upsert_qa_pairs([_qa(1)])
        stats = kb.stats()
    assert stats["documents"] == 2
    assert stats["qa_pairs"] == 1
    assert stats["documents_by_kind"]["issue"] == 2
