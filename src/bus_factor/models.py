"""Core data types shared across the pipeline.

Design notes
------------
* Everything is a plain ``@dataclass`` with explicit ``to_dict``/``from_dict``
  so records round-trip through JSONL without a serialization framework.
* ``Source`` is deliberately separate from ``Document``: one source (an issue)
  can spawn several documents (body, resolution comment), and every answer
  must be able to point back at the *source*, not just the chunk it retrieved.
* Timestamps are stored as ISO-8601 strings and parsed on demand. Staleness is
  first-class here because knowledge decays and a trustworthy answer has to say
  how old its evidence is.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from bus_factor.errors import DataError


def _require(d: Any, keys: tuple[str, ...], ctx: str) -> None:
    """Validate that ``d`` is a mapping containing every required key."""
    if not isinstance(d, dict):
        raise DataError(f"{ctx}: expected an object, got {type(d).__name__}")
    missing = [k for k in keys if k not in d]
    if missing:
        raise DataError(f"{ctx}: missing required field(s): {', '.join(missing)}")


def parse_iso(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp, tolerating a trailing ``Z``."""
    if not value:
        return None
    cleaned = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


@dataclass(frozen=True)
class Source:
    """Where a piece of knowledge came from. Immutable so citations are stable."""

    id: str
    kind: str  # "issue" | "pr" | "doc" | "commit" | "interview"
    title: str
    url: str = ""
    author: str = ""
    created_at: str = ""  # ISO-8601

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Source:
        _require(d, ("id", "kind", "title"), "Source")
        return cls(**{k: d.get(k, "") for k in cls.__dataclass_fields__})


@dataclass
class Document:
    """A retrievable unit of knowledge with a pointer back to its ``Source``."""

    id: str
    text: str
    source: Source
    metadata: dict[str, Any] = field(default_factory=dict)

    def age_days(self, as_of: date | None = None) -> float | None:
        """Age of the underlying source in days, or ``None`` if undated."""
        created = parse_iso(self.source.created_at)
        if created is None:
            return None
        reference = as_of or date.today()
        return max(0.0, (reference - created.date()).days)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Document:
        _require(d, ("id", "text", "source"), "Document")
        return cls(
            id=d["id"],
            text=d["text"],
            source=Source.from_dict(d["source"]),
            metadata=d.get("metadata", {}),
        )


@dataclass
class QAPair:
    """A ground-truth question/answer used for evaluation.

    ``reference_answer`` is what the human authority actually said. The whole
    project is judged on how well the system reproduces these under holdout.
    """

    id: str
    question: str
    reference_answer: str
    source: Source
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "source": self.source.to_dict(),
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QAPair:
        _require(d, ("id", "question", "reference_answer", "source"), "QAPair")
        return cls(
            id=d["id"],
            question=d["question"],
            reference_answer=d["reference_answer"],
            source=Source.from_dict(d["source"]),
            tags=d.get("tags", []),
        )


@dataclass
class Citation:
    """A single evidence pointer attached to an answer."""

    document_id: str
    title: str
    url: str
    score: float
    age_days: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Answer:
    """The system's response. Never bare text — always provenance-carrying.

    ``confidence`` is derived from retrieval quality, and ``staleness_days`` is
    the age of the freshest cited evidence. Together they let a caller decide
    whether to trust the answer or route to a human.
    """

    text: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    staleness_days: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "citations": [c.to_dict() for c in self.citations],
            "confidence": self.confidence,
            "staleness_days": self.staleness_days,
            "meta": self.meta,
        }


# --- JSONL helpers -----------------------------------------------------------


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> int:
    """Write dataclass-like rows (anything with ``to_dict``) to JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            payload = row.to_dict() if hasattr(row, "to_dict") else row
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream raw dict rows from a JSONL file."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
