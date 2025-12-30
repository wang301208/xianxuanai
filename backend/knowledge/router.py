"""Routing logic between short-term vector memory and long-term concept graph."""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore

from backend.knowledge.vector_store import LocalVectorStore
from backend.knowledge.consolidation import KnowledgeConsolidator
from collections import OrderedDict
import os


def _hash_embedding(text: str, dimensions: int = 12) -> List[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    chunk = max(1, len(digest) // dimensions)
    embedding: List[float] = []
    for index in range(0, len(digest), chunk):
        piece = digest[index : index + chunk]
        if not piece:
            continue
        integer = int.from_bytes(piece, byteorder="big", signed=False)
        embedding.append(integer / float(256 ** len(piece)))
        if len(embedding) == dimensions:
            break
    while len(embedding) < dimensions:
        embedding.append(0.0)
    return embedding[:dimensions]


class MemoryRouter:
    """Coordinate short-term vector storage and long-term knowledge consolidation."""

    def __init__(
        self,
        consolidator: KnowledgeConsolidator,
        *,
        sentence_model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._consolidator = consolidator
        self._sentence_model: Optional[SentenceTransformer] = None
        self._sentence_model_name = sentence_model_name
        self._store: Optional[LocalVectorStore] = None
        self._entries: Dict[str, Dict[str, object]] = {}
        self._embed_cache: OrderedDict[str, List[float]] = OrderedDict()
        self._embed_cache_size = int(os.getenv("MEMORY_EMBED_CACHE_SIZE", "512"))

    # ------------------------------------------------------------------#
    def add_observation(
        self,
        text: str,
        *,
        source: str,
        metadata: Optional[Dict[str, object]] = None,
        promote: bool = False,
    ) -> str:
        """Add ``text`` to short-term memory and optionally promote immediately."""

        text = text.strip()
        if not text:
            return ""

        if promote:
            self._consolidator.record_statement(text, source=source, metadata=metadata or {})
            return ""

        entry_id = uuid.uuid4().hex[:12]
        embedding = self._encode(text)

        if self._store is None:
            self._store = LocalVectorStore(len(embedding), use_faiss=False)

        self._store.add(
            embedding,
            {
                "id": entry_id,
                "source": source,
            },
        )
        self._entries[entry_id] = {
            "text": text,
            "source": source,
            "metadata": metadata or {},
            "created_at": time.time(),
            "last_accessed": None,
            "usage": 0,
            "promoted": False,
            "embedding": embedding,
        }
        return entry_id

    # ------------------------------------------------------------------#
    def query(
        self,
        text: str,
        *,
        top_k: int = 5,
    ) -> List[Dict[str, object]]:
        """Return the most similar stored entries and mark them as used."""

        if not self._store or not self._entries:
            return []
        embedding = self._encode(text)
        results = self._store.search(embedding, top_k=top_k)

        matches: List[Dict[str, object]] = []
        for hit in results:
            entry_id = hit.get("id")
            if not entry_id:
                continue
            entry = self._entries.get(entry_id)
            if not entry:
                continue
            entry["usage"] = int(entry["usage"]) + 1
            entry["last_accessed"] = time.time()
            try:
                from backend.monitoring import record_memory_hit

                record_memory_hit()
            except Exception:
                pass
            matches.append(
                {
                    "id": entry_id,
                    "text": entry["text"],
                    "source": entry["source"],
                    "metadata": entry["metadata"],
                    "similarity": hit.get("similarity"),
                }
            )
        return matches

    # ------------------------------------------------------------------#
    def review(
        self,
        *,
        usage_threshold: int = 3,
        max_age: float = 900.0,
    ) -> List[str]:
        """Promote entries that meet the usage/age criteria."""

        promoted: List[str] = []
        now = time.time()
        for entry_id, entry in list(self._entries.items()):
            if entry["promoted"]:
                continue
            usage = int(entry.get("usage", 0))
            age = now - float(entry.get("created_at", now))
            if usage >= usage_threshold or (age >= max_age and usage > 0):
                self._consolidator.record_statement(
                    entry["text"],
                    source=str(entry["source"]),
                    metadata={"promotion": True, **entry["metadata"]},
                )
                entry["promoted"] = True
                promoted.append(entry_id)
        return promoted

    # ------------------------------------------------------------------#
    def stats(self) -> Dict[str, int]:
        return {
            "total_entries": len(self._entries),
            "promoted": sum(1 for e in self._entries.values() if e.get("promoted")),
        }

    # ------------------------------------------------------------------#
    def shrink(
        self,
        *,
        max_entries: Optional[int] = None,
        max_age: Optional[float] = None,
        min_usage: int = 0,
    ) -> int:
        """Discard stale entries and rebuild the vector index.

        Parameters
        ----------
        max_entries:
            Optional cap on how many entries to keep. When provided, the least
            used and oldest items beyond this cap are removed.
        max_age:
            Optional maximum age (seconds). Entries older than this and with
            usage less than or equal to ``min_usage`` are dropped.
        min_usage:
            Minimum usage count required for an entry to bypass the ``max_age``
            filter.
        """

        if not self._entries:
            return 0

        now = time.time()
        survivors: List[tuple[str, Dict[str, object]]] = []
        for entry_id, entry in self._entries.items():
            created = float(entry.get("created_at", now))
            age = now - created
            usage = int(entry.get("usage", 0) or 0)
            if max_age is not None and age > max_age and usage <= min_usage:
                continue
            survivors.append((entry_id, entry))

        if max_entries is not None and len(survivors) > max_entries:
            survivors.sort(
                key=lambda item: (
                    int(item[1].get("usage", 0) or 0),
                    float(item[1].get("created_at", 0.0) or 0.0),
                )
            )
            survivors = survivors[-max_entries:]

        removed = len(self._entries) - len(survivors)
        if removed <= 0:
            return 0

        self._entries = {entry_id: entry for entry_id, entry in survivors}
        self._rebuild_store()
        return removed

    # ------------------------------------------------------------------#
    def _encode(self, text: str) -> List[float]:
        if not text:
            return []
        cached = self._embed_cache.get(text)
        if cached is not None:
            self._embed_cache.move_to_end(text)
            return list(cached)

        model = self._ensure_model()
        if model is not None:
            try:
                vector = model.encode(text, convert_to_numpy=True)
                embedding = vector.astype(float).tolist()
            except Exception:  # pragma: no cover - graceful fallback
                embedding = _hash_embedding(text)
        else:
            embedding = _hash_embedding(text)

        self._embed_cache[text] = embedding
        if len(self._embed_cache) > self._embed_cache_size > 0:
            self._embed_cache.popitem(last=False)
        return list(embedding)

    def _ensure_model(self) -> Optional[SentenceTransformer]:
        if self._sentence_model is not None or SentenceTransformer is None:
            return self._sentence_model
        try:
            self._sentence_model = SentenceTransformer(self._sentence_model_name)
        except Exception:  # pragma: no cover - optional dependency
            self._sentence_model = None
        return self._sentence_model

    def _rebuild_store(self) -> None:
        if not self._entries:
            self._store = None
            return

        sample_embedding: Optional[List[float]] = None
        for entry in self._entries.values():
            embedding = entry.get("embedding")
            if isinstance(embedding, list):
                sample_embedding = embedding
                break

        if not sample_embedding:
            self._store = None
            return

        self._store = LocalVectorStore(len(sample_embedding), use_faiss=False)
        for entry_id, entry in self._entries.items():
            embedding = entry.get("embedding")
            if not isinstance(embedding, list):
                continue
            self._store.add(
                embedding,
                {
                    "id": entry_id,
                    "source": entry.get("source"),
                },
            )
