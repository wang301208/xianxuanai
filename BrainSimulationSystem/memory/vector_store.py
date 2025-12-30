"""Optional vector-memory integration backed by Chroma."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:  # pragma: no cover - optional dependency
    from backend.forge.forge.memory.chroma_memstore import ChromaMemStore
except Exception:  # pragma: no cover
    ChromaMemStore = None  # type: ignore[assignment]


def _to_serializable_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(content)


class VectorMemoryStore:
    """Wrapper that exposes a minimal interface for semantic storage/retrieval."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, *, logger: Optional[logging.Logger] = None) -> None:
        cfg = config or {}
        self.enabled: bool = bool(cfg.get("enabled", False))
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        self._backend = (cfg.get("backend") or "chroma").lower()
        self.top_k = int(cfg.get("top_k", 8))
        self.collection = cfg.get("collection", "cognitive_memory")
        self._store: Optional[ChromaMemStore] = None
        self._available = False

        if not self.enabled:
            return

        if self._backend == "chroma":
            if ChromaMemStore is None:
                self._logger.warning("ChromaMemStore is unavailable; disable vector_store or install chromadb.")
                self.enabled = False
                return
            path = cfg.get("path", "BrainSimulationSystem/data/vector_memory")
            try:
                self._store = ChromaMemStore(path)
                self._available = True
            except Exception as exc:  # pragma: no cover - chroma runtime failure
                self._logger.warning("Failed to initialize ChromaMemStore at %s: %s", path, exc)
                self.enabled = False
        else:
            self._logger.warning("Unknown vector memory backend '%s'; disabling vector store.", self._backend)
            self.enabled = False

    @property
    def is_available(self) -> bool:
        return self.enabled and self._available and self._store is not None

    def index_memory(
        self,
        memory_id: Optional[str],
        content: Any,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.is_available:
            return
        document = _to_serializable_text(content)
        meta = dict(metadata or {})
        if memory_id is not None:
            meta.setdefault("memory_id", str(memory_id))
        try:
            self._store.add(self.collection, document, meta)
        except Exception as exc:  # pragma: no cover - runtime safety
            self._logger.debug("Vector store add failed: %s", exc)

    def similarity_search(
        self,
        query_text: str,
        *,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.is_available or not query_text:
            return []
        try:
            response = self._store.query(
                self.collection,
                query_text,
                filters=filters,
                document_search=None,
            )
        except Exception as exc:  # pragma: no cover
            self._logger.debug("Vector store query failed: %s", exc)
            return []

        matches: List[Dict[str, Any]] = []
        ids: Sequence[str] = response.get("ids", [[]])[0] if response else []
        docs: Sequence[str] = response.get("documents", [[]])[0] if response else []
        metas: Sequence[Any] = response.get("metadatas", [[]])[0] if response else []
        distances: Sequence[float] = response.get("distances", [[]])[0] if response else []

        limit = top_k or self.top_k
        for idx, (doc_id, doc, meta, distance) in enumerate(zip(ids, docs, metas, distances)):
            if idx >= limit:
                break
            if not isinstance(meta, dict):
                meta = {"value": meta}
            score = 1.0 / (1.0 + float(distance)) if distance is not None else None
            matches.append(
                {
                    "id": doc_id,
                    "content": doc,
                    "metadata": meta or {},
                    "score": score,
                    "distance": float(distance) if distance is not None else None,
                }
            )
        return matches

    def flush_records(self, identifiers: Iterable[str]) -> None:
        if not self.is_available:
            return
        store = self._store
        assert store is not None
        for identifier in identifiers:
            try:
                store.delete(self.collection, str(identifier))
            except Exception as exc:  # pragma: no cover
                self._logger.debug("Vector store delete failed for %s: %s", identifier, exc)
