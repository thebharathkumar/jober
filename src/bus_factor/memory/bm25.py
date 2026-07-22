"""A small, dependency-free BM25 Okapi ranker.

Why hand-roll it: the whole point of the core path is that it runs anywhere
with no install step. BM25 is ~40 lines and well understood, so bundling it
removes the one dependency that a lexical baseline would otherwise need. When
``rank-bm25`` is installed the store can swap to it, but results are equivalent
for the corpus sizes this project targets.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """Classic BM25 Okapi over an in-memory corpus of token lists."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.n = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.doc_freqs: list[Counter[str]] = [Counter(doc) for doc in corpus]

        # Document frequency per term, then smoothed IDF.
        df: Counter[str] = Counter()
        for freqs in self.doc_freqs:
            df.update(freqs.keys())
        self.idf: dict[str, float] = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def scores(self, query: str) -> list[float]:
        """BM25 score of the query against every document (index-aligned)."""
        q_terms = tokenize(query)
        out = [0.0] * self.n
        for i in range(self.n):
            freqs = self.doc_freqs[i]
            dl = self.doc_len[i]
            denom_norm = self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            total = 0.0
            for term in q_terms:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                total += idf * (tf * (self.k1 + 1)) / (tf + denom_norm)
            out[i] = total
        return out

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        ranked = sorted(enumerate(self.scores(query)), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in ranked[:k] if s > 0.0]
