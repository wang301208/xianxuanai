"""Lightweight local vector store for semantic search.

This module implements a minimal wrapper around FAISS_ for storing and
retrieving embedding vectors.  If FAISS is not available, it falls back to a
simple numpy based index.  The implementation is intentionally compact yet
sufficient for small projects and unit tests.

Usage
-----
>>> store = LocalVectorStore(3)
>>> store.add([0.1, 0.2, 0.3], {"id": 1})
>>> store.search([0.1, 0.2, 0.3])
[{"id": 1}]

.. _FAISS: https://github.com/facebookresearch/faiss
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

try:
    import numpy as np  # type: ignore
    _NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore
    _NUMPY_AVAILABLE = False

try:  # Optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover - handled gracefully
    faiss = None  # type: ignore


@dataclass
class LocalVectorStore:
    """Small in-memory vector index with optional FAISS acceleration."""

    dimension: int
    use_faiss: bool = True
    _faiss_index: Any | None = field(init=False, default=None)
    _embeddings: List[np.ndarray] = field(init=False, default_factory=list)
    _meta: List[Dict[str, Any]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.use_faiss = self.use_faiss and faiss is not None and _NUMPY_AVAILABLE
        if self.use_faiss:
            # Inner product is equivalent to cosine similarity on normalised vectors
            self._faiss_index = faiss.IndexFlatIP(self.dimension)

    # ------------------------------------------------------------------
    def add(self, vector: Iterable[float], metadata: Dict[str, Any]) -> None:
        """Add a vector with associated metadata to the store."""

        if not _NUMPY_AVAILABLE:
            vec_list = [float(v) for v in vector]
            if len(vec_list) != self.dimension:
                raise ValueError("Vector has wrong dimensions")
            self._embeddings.append(vec_list)
        else:
            vec = np.asarray(vector, dtype="float32")
            if vec.shape != (self.dimension,):
                raise ValueError("Vector has wrong dimensions")
            if self.use_faiss and self._faiss_index is not None:
                self._faiss_index.add(vec.reshape(1, -1))
            else:
                self._embeddings.append(vec)
        self._meta.append(metadata)

    # ------------------------------------------------------------------
    def search(self, query: Iterable[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Return metadata for vectors most similar to ``query``."""

        if not _NUMPY_AVAILABLE:
            q = [float(v) for v in query]
            if len(q) != self.dimension:
                raise ValueError("Query has wrong dimensions")
            if not self._embeddings:
                return []
            sims: List[float] = []
            q_norm = math.sqrt(sum(val * val for val in q)) or 1e-10
            for emb in self._embeddings:
                emb_norm = math.sqrt(sum(val * val for val in emb)) or 1e-10
                dot = sum(e * qv for e, qv in zip(emb, q))
                sims.append(dot / (emb_norm * q_norm))
            best_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
            results: List[Dict[str, Any]] = []
            for index in best_idx:
                meta = dict(self._meta[index])
                meta["similarity"] = float(sims[index])
                results.append(meta)
            return results

        q = np.asarray(query, dtype="float32")
        if q.shape != (self.dimension,):
            raise ValueError("Query has wrong dimensions")

        if self.use_faiss and self._faiss_index is not None and self._faiss_index.ntotal > 0:
            distances, idx = self._faiss_index.search(q.reshape(1, -1), top_k)
            results: List[Dict[str, Any]] = []
            for score, index in zip(distances[0], idx[0]):
                if index < 0 or index >= len(self._meta):
                    continue
                meta = dict(self._meta[int(index)])
                meta["similarity"] = float(score)
                results.append(meta)
            return results

        if not self._embeddings:
            return []
        embs = np.vstack(self._embeddings)
        norms = np.linalg.norm(embs, axis=1) * np.linalg.norm(q)
        norms = np.where(norms == 0, 1e-10, norms)
        sims = embs @ q / norms
        best_idx = np.argsort(sims)[::-1][:top_k]
        results: List[Dict[str, Any]] = []
        for index in best_idx:
            meta = dict(self._meta[int(index)])
            meta["similarity"] = float(sims[int(index)])
            results.append(meta)
        return results
