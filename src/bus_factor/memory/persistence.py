"""SQLite-backed persistent knowledge base.

The in-memory ``MemoryStore`` is the retrieval *index*; this is the durable
*source of truth*. Separating them matters: you ingest expensively once (network,
rate limits), persist here, and rebuild the cheap in-memory index from SQLite on
every start. Ingest is incremental — documents upsert by id, so re-running
against a repo updates only what changed instead of re-scraping everything.

SQLite is intentional: zero-ops, single-file, ACID, and already in the standard
library. It scales comfortably to the corpus sizes a single team's knowledge
occupies, and swapping in Postgres later is a matter of this one module.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import TracebackType

from bus_factor.logging_config import get_logger
from bus_factor.models import Document, QAPair, Source

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id                TEXT PRIMARY KEY,
    text              TEXT NOT NULL,
    source_id         TEXT NOT NULL,
    source_kind       TEXT NOT NULL,
    source_title      TEXT NOT NULL,
    source_url        TEXT NOT NULL,
    source_author     TEXT NOT NULL,
    source_created_at TEXT NOT NULL,
    metadata_json     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_kind ON documents (source_kind);

CREATE TABLE IF NOT EXISTS qa_pairs (
    id                TEXT PRIMARY KEY,
    question          TEXT NOT NULL,
    reference_answer  TEXT NOT NULL,
    source_id         TEXT NOT NULL,
    source_kind       TEXT NOT NULL,
    source_title      TEXT NOT NULL,
    source_url        TEXT NOT NULL,
    source_author     TEXT NOT NULL,
    source_created_at TEXT NOT NULL,
    tags_json         TEXT NOT NULL
);
"""


class SqliteKnowledgeBase:
    """Durable store for knowledge documents and ground-truth QA pairs."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # --- writes (incremental upsert) ---------------------------------------

    def upsert_documents(self, docs: list[Document]) -> int:
        rows = [
            (
                d.id,
                d.text,
                d.source.id,
                d.source.kind,
                d.source.title,
                d.source.url,
                d.source.author,
                d.source.created_at,
                json.dumps(d.metadata, ensure_ascii=False),
            )
            for d in docs
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO documents "
            "(id, text, source_id, source_kind, source_title, source_url, "
            " source_author, source_created_at, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        log.info("upserted %d documents into %s", len(rows), self.path)
        return len(rows)

    def upsert_qa_pairs(self, pairs: list[QAPair]) -> int:
        rows = [
            (
                p.id,
                p.question,
                p.reference_answer,
                p.source.id,
                p.source.kind,
                p.source.title,
                p.source.url,
                p.source.author,
                p.source.created_at,
                json.dumps(p.tags, ensure_ascii=False),
            )
            for p in pairs
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO qa_pairs "
            "(id, question, reference_answer, source_id, source_kind, source_title, "
            " source_url, source_author, source_created_at, tags_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    # --- reads --------------------------------------------------------------

    def documents(self) -> list[Document]:
        cur = self._conn.execute("SELECT * FROM documents")
        return [self._row_to_document(r) for r in cur.fetchall()]

    def qa_pairs(self) -> list[QAPair]:
        cur = self._conn.execute("SELECT * FROM qa_pairs")
        return [self._row_to_qa(r) for r in cur.fetchall()]

    def stats(self) -> dict[str, object]:
        n_docs = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        n_qa = self._conn.execute("SELECT COUNT(*) FROM qa_pairs").fetchone()[0]
        by_kind = {
            row["source_kind"]: row["n"]
            for row in self._conn.execute(
                "SELECT source_kind, COUNT(*) AS n FROM documents GROUP BY source_kind"
            ).fetchall()
        }
        return {"documents": n_docs, "qa_pairs": n_qa, "documents_by_kind": by_kind}

    # --- helpers ------------------------------------------------------------

    @staticmethod
    def _row_to_source(row: sqlite3.Row) -> Source:
        return Source(
            id=row["source_id"],
            kind=row["source_kind"],
            title=row["source_title"],
            url=row["source_url"],
            author=row["source_author"],
            created_at=row["source_created_at"],
        )

    @classmethod
    def _row_to_document(cls, row: sqlite3.Row) -> Document:
        return Document(
            id=row["id"],
            text=row["text"],
            source=cls._row_to_source(row),
            metadata=json.loads(row["metadata_json"]),
        )

    @classmethod
    def _row_to_qa(cls, row: sqlite3.Row) -> QAPair:
        return QAPair(
            id=row["id"],
            question=row["question"],
            reference_answer=row["reference_answer"],
            source=cls._row_to_source(row),
            tags=json.loads(row["tags_json"]),
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SqliteKnowledgeBase:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
