"""Optional dense retriever. Imported lazily so the core never needs it.

If ``sentence-transformers`` / ``numpy`` are not installed, ``DenseRetriever``
reports ``available = False`` and the store silently runs lexical-only. This is
the honest way to ship an optional accelerator: the feature degrades, it does
not crash.
"""

from __future__ import annotations

from typing import Any


class DenseRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # Typed Any because the concrete types come from optional dependencies
        # that may not be importable; the `available` flag gates all real use.
        self.available = False
        self._model: Any = None
        self._np: Any = None
        self._matrix: Any = None
        try:
            import numpy as np  # noqa: PLC0415
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._np = np
            self._model = SentenceTransformer(model_name)
            self.available = True
        except Exception:
            # Any failure (missing package, no model cache, offline) -> disabled.
            self.available = False

    def index(self, texts: list[str]) -> None:
        if not self.available:
            return
        self._matrix = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        if not self.available or self._matrix is None:
            return []
        q = self._model.encode([query], normalize_embeddings=True)[0]
        sims = self._matrix @ q  # cosine, vectors are normalized
        order = self._np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]
