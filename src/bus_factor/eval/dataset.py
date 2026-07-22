"""Load eval data and split a deterministic holdout.

The split is by a stable hash of each QA id, not by random shuffle. That means
the holdout is identical on every machine and every run without threading a seed
around — a property you want when a metric has to be reproducible in CI and
comparable across commits.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from bus_factor.errors import DataError
from bus_factor.logging_config import get_logger
from bus_factor.models import Document, QAPair, read_jsonl

log = get_logger(__name__)

T = TypeVar("T")


def _load_resilient(path: str | Path, parse: Callable[[dict], T], kind: str) -> list[T]:
    """Parse a JSONL file row by row, skipping (and logging) malformed rows.

    One corrupt line in a 10k-row export should not abort the whole load — an
    enterprise pipeline reports the bad rows and proceeds with the good ones.
    """
    rows: list[T] = []
    skipped = 0
    for i, raw in enumerate(read_jsonl(path)):
        try:
            rows.append(parse(raw))
        except DataError as exc:
            skipped += 1
            log.warning("%s: skipping malformed %s at line %d: %s", path, kind, i + 1, exc)
    if skipped:
        log.warning("%s: loaded %d %s, skipped %d malformed", path, len(rows), kind, skipped)
    return rows


def load_qa_pairs(path: str | Path) -> list[QAPair]:
    return _load_resilient(path, QAPair.from_dict, "QAPair")


def load_documents(path: str | Path) -> list[Document]:
    return _load_resilient(path, Document.from_dict, "Document")


def _bucket(qa_id: str) -> float:
    """Map an id to a stable value in [0, 1)."""
    digest = hashlib.sha256(qa_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def split_holdout(
    pairs: list[QAPair], holdout_fraction: float = 0.3
) -> tuple[list[QAPair], list[QAPair]]:
    """Return ``(train, holdout)`` deterministically by id hash."""
    train: list[QAPair] = []
    holdout: list[QAPair] = []
    for p in pairs:
        (holdout if _bucket(p.id) < holdout_fraction else train).append(p)
    return train, holdout
