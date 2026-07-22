"""Schema validation on data load: bad rows are rejected or skipped, not crashed on."""

from __future__ import annotations

import json

import pytest

from bus_factor.errors import DataError
from bus_factor.eval.dataset import load_documents
from bus_factor.models import Document, QAPair, Source


def test_source_missing_required_field_raises():
    with pytest.raises(DataError):
        Source.from_dict({"id": "x"})  # missing kind, title


def test_document_missing_source_raises():
    with pytest.raises(DataError):
        Document.from_dict({"id": "x", "text": "y"})


def test_document_non_object_raises():
    with pytest.raises(DataError):
        Document.from_dict(["not", "a", "dict"])  # type: ignore[arg-type]


def test_qapair_roundtrips_through_dict():
    qa = QAPair(
        id="q1",
        question="q",
        reference_answer="a",
        source=Source(id="s1", kind="issue", title="t"),
    )
    assert QAPair.from_dict(qa.to_dict()).id == "q1"


def test_load_documents_skips_malformed_rows(tmp_path):
    good = Document(id="d1", text="t", source=Source(id="s1", kind="issue", title="ti"))
    path = tmp_path / "docs.jsonl"
    path.write_text(
        json.dumps(good.to_dict()) + "\n" + json.dumps({"id": "bad"}) + "\n",
        encoding="utf-8",
    )
    docs = load_documents(path)
    assert len(docs) == 1
    assert docs[0].id == "d1"
