"""Load eval data and split a deterministic holdout.

The split is by a stable hash of each QA id, not by random shuffle. That means
the holdout is identical on every machine and every run without threading a seed
around — a property you want when a metric has to be reproducible in CI and
comparable across commits.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from bus_factor.models import Document, QAPair, read_jsonl


def load_qa_pairs(path: str | Path) -> list[QAPair]:
    return [QAPair.from_dict(row) for row in read_jsonl(path)]


def load_documents(path: str | Path) -> list[Document]:
    return [Document.from_dict(row) for row in read_jsonl(path)]


def _bucket(qa_id: str) -> float:
    """Map an id to a stable value in [0, 1)."""
    digest = hashlib.sha256(qa_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def split_holdout(
    pairs: list[QAPair], holdout_fraction: float = 0.3
) -> tuple[list[QAPair], list[QAPair]]:
    """Return ``(train, holdout)`` deterministically by id hash."""
    train, holdout = [], []
    for p in pairs:
        (holdout if _bucket(p.id) < holdout_fraction else train).append(p)
    return train, holdout
