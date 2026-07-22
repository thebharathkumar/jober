"""Retrieval-layer tests: BM25 ranking and the hybrid store."""

from __future__ import annotations

from bus_factor.memory.bm25 import BM25, tokenize
from bus_factor.memory.store import MemoryStore
from bus_factor.models import Document, Source


def _doc(doc_id: str, title: str, text: str) -> Document:
    return Document(id=doc_id, text=text, source=Source(id=doc_id, kind="doc", title=title))


def test_bm25_ranks_relevant_document_first():
    corpus = [
        tokenize("installing on apple silicon requires an arm64 wheel"),
        tokenize("configuring the cache directory with an environment variable"),
        tokenize("streaming large csv files in constant memory"),
    ]
    bm25 = BM25(corpus)
    top = bm25.top_k("how to install on apple silicon", k=1)
    assert top and top[0][0] == 0


def test_store_search_returns_provenance_and_ordering():
    store = MemoryStore()
    store.add(
        [
            _doc("d1", "Apple Silicon install", "arm64 wheel install on apple silicon"),
            _doc("d2", "Cache config", "set the cache directory environment variable"),
        ]
    )
    results = store.search("apple silicon install", k=2)
    assert results
    assert results[0].document.id == "d1"
    assert results[0].rank == 0
    assert results[0].document.source.title == "Apple Silicon install"


def test_store_empty_returns_no_results():
    assert MemoryStore().search("anything", k=3) == []
