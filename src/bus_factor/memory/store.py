"""The retrieval store: documents in, ranked evidence out.

Retrieval is hybrid by design. Lexical BM25 catches exact identifiers (error
strings, function names, flags) that dense models blur; dense catches paraphrase
that lexical misses. We fuse them with Reciprocal Rank Fusion (RRF), which needs
no score calibration between the two very different scales — a property that
matters because BM25 scores and cosine similarities are not comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from bus_factor.memory.bm25 import BM25
from bus_factor.models import Document


@dataclass
class RetrievalResult:
    document: Document
    score: float  # fused, higher is better
    rank: int  # 0-based
    lexical_score: float = 0.0


class MemoryStore:
    """In-memory hybrid index over ``Document`` objects."""

    def __init__(self, rrf_k: int = 60, use_dense: bool = False) -> None:
        self.rrf_k = rrf_k
        self.documents: list[Document] = []
        self._bm25: BM25 | None = None
        self._dense = None
        self._use_dense = use_dense

    def add(self, docs: list[Document]) -> None:
        self.documents.extend(docs)
        self._reindex()

    def _reindex(self) -> None:
        texts = [self._doc_text(d) for d in self.documents]
        self._bm25 = BM25([t.split() for t in texts]) if self.documents else None
        if self._use_dense:
            from bus_factor.memory.embeddings import DenseRetriever

            self._dense = DenseRetriever()
            if self._dense.available:
                self._dense.index(texts)
            else:
                self._dense = None

    @staticmethod
    def _doc_text(doc: Document) -> str:
        # Title carries strong signal for issue-style content, so weight it in.
        return f"{doc.source.title}\n{doc.source.title}\n{doc.text}".lower()

    def search(self, query: str, k: int = 5, as_of: date | None = None) -> list[RetrievalResult]:
        if not self.documents or self._bm25 is None:
            return []

        # Rank lists from each retriever: list of (doc_index, score).
        pool = max(k * 4, 20)
        lexical = self._bm25.top_k(query, pool)
        lexical_by_idx = dict(lexical)
        dense = self._dense.top_k(query, pool) if self._dense is not None else []

        fused = self._rrf([lexical, dense])
        results: list[RetrievalResult] = []
        for rank, (idx, fused_score) in enumerate(fused[:k]):
            results.append(
                RetrievalResult(
                    document=self.documents[idx],
                    score=fused_score,
                    rank=rank,
                    lexical_score=lexical_by_idx.get(idx, 0.0),
                )
            )
        return results

    def _rrf(self, rank_lists: list[list[tuple[int, float]]]) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion across one or more ranked lists."""
        fused: dict[int, float] = {}
        for rank_list in rank_lists:
            for rank, (idx, _score) in enumerate(rank_list):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)
