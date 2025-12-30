"""Persistent long-term memory wrapper for agent learning.

This module introduces a light-weight `KnowledgeBase` interface focused on
storing and retrieving experience, reflections, and reusable rules.

Design goals:
- Local-first persistence via SQLite (`backend.memory.LongTermMemory`)
- Optional semantic recall via deterministic embeddings (hashing)
- Simple, structured API (`save_memory`, `query_memory`)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    from backend.memory import LongTermMemory
except Exception:  # pragma: no cover - fallback when backend stack unavailable
    LongTermMemory = None  # type: ignore[assignment]

try:
    from modules.memory.embedders import HashingEmbedder
except Exception:  # pragma: no cover - minimal fallback
    HashingEmbedder = None  # type: ignore[assignment]


EmbeddingFn = Callable[[str], Sequence[float]]


def _default_db_path() -> Path:
    raw = os.getenv("KNOWLEDGE_BASE_DB_PATH") or os.getenv("LONG_TERM_MEMORY_PATH") or "data/memory.db"
    return Path(raw)


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_vector(vector: Sequence[float]) -> List[float]:
    try:
        return [float(v) for v in vector]
    except Exception:
        return []


@dataclass(frozen=True)
class KnowledgeBaseItem:
    """A stored memory item (optionally with an attached similarity score)."""

    id: int
    category: str
    content: str
    timestamp: float
    tags: List[str] = field(default_factory=list)
    confidence: float = 1.0
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float | None = None


class KnowledgeBase:
    """Unified long-term memory store with optional semantic search."""

    def __init__(
        self,
        *,
        db_path: str | Path | None = None,
        memory: Any | None = None,
        embedder: EmbeddingFn | None = None,
        enabled: bool | None = None,
        embedding_enabled: bool | None = None,
    ) -> None:
        self.enabled = _parse_bool(os.getenv("KNOWLEDGE_BASE_ENABLED"), default=True) if enabled is None else bool(enabled)
        self.embedding_enabled = (
            _parse_bool(os.getenv("KNOWLEDGE_BASE_EMBEDDINGS_ENABLED"), default=True)
            if embedding_enabled is None
            else bool(embedding_enabled)
        )
        self.db_path = Path(db_path) if db_path is not None else _default_db_path()

        if memory is not None:
            self._memory = memory
        else:
            if LongTermMemory is None:
                raise RuntimeError("backend.memory.LongTermMemory is unavailable")
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._memory = LongTermMemory(self.db_path)

        if embedder is not None:
            self._embedder = embedder
        else:
            dimension_raw = os.getenv("KNOWLEDGE_BASE_EMBEDDING_DIM", "384")
            try:
                dimension = int(dimension_raw)
            except ValueError:
                dimension = 384
            if HashingEmbedder is None:
                self._embedder = lambda text: []  # type: ignore[assignment]
                self.embedding_enabled = False
            else:
                self._embedder = HashingEmbedder(dimension=dimension)

    @classmethod
    def from_env(cls) -> "KnowledgeBase":
        return cls()

    @property
    def memory(self) -> Any:
        return self._memory

    def close(self) -> None:
        if hasattr(self._memory, "close"):
            try:
                self._memory.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ public API
    def save_memory(
        self,
        category: str,
        content: str,
        *,
        tags: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        confidence: Optional[float] = None,
        status: str = "active",
        embed: bool | None = None,
    ) -> int | None:
        """Persist `content` and (optionally) index it for similarity search."""

        if not self.enabled:
            return None

        timestamp = time.time()
        tags_list = [str(tag) for tag in (tags or []) if str(tag).strip()]
        entry_id = int(
            self._memory.add(
                str(category or "general"),
                str(content or "").strip(),
                tags=tags_list or None,
                timestamp=timestamp,
                confidence=confidence,
                status=status,
                metadata=dict(metadata or {}),
            )
        )

        should_embed = self.embedding_enabled if embed is None else bool(embed)
        if should_embed:
            vector = _coerce_vector(self._embedder(str(content or "")))
            if vector:
                try:
                    self._memory.add_embedding(
                        str(entry_id),
                        vector,
                        metadata={
                            "memory_id": entry_id,
                            "category": str(category or "general"),
                            "tags": sorted(tags_list),
                            "timestamp": timestamp,
                        },
                    )
                except Exception:
                    pass

        return entry_id

    def query_memory(
        self,
        query: str,
        *,
        top_k: int = 5,
        category: str | None = None,
        tags: Optional[Sequence[str]] = None,
        allow_fallback_search: bool = True,
    ) -> List[KnowledgeBaseItem]:
        """Return memories most relevant to `query`."""

        if not self.enabled:
            return []

        query = str(query or "").strip()
        if not query:
            return []

        hits: List[KnowledgeBaseItem] = []
        if self.embedding_enabled:
            vector = _coerce_vector(self._embedder(query))
            if vector:
                try:
                    results = self._memory.similarity_search(
                        vector,
                        top_k=max(1, int(top_k)),
                        category=category,
                        tags=list(tags or []) if tags else None,
                    )
                except Exception:
                    results = []
                for result in results:
                    row = result.get("memory") or {}
                    if not isinstance(row, dict):
                        continue
                    hits.append(
                        KnowledgeBaseItem(
                            id=int(row.get("id") or result.get("memory_id") or 0),
                            category=str(row.get("category") or category or "general"),
                            content=str(row.get("content") or ""),
                            timestamp=float(row.get("timestamp") or 0.0),
                            tags=list(row.get("tags") or []),
                            confidence=float(row.get("confidence") or 1.0),
                            status=str(row.get("status") or "active"),
                            metadata=dict(row.get("metadata") or {}),
                            score=float(result.get("score") or 0.0),
                        )
                    )
        if hits or not allow_fallback_search:
            return hits[: max(1, int(top_k))]

        # Fallback to LIKE-based search over the raw memory table.
        try:
            matches = self._memory.search(query, limit=max(1, int(top_k)), category=category)
        except Exception:
            matches = []
        for match in matches:
            if not isinstance(match, dict):
                continue
            hits.append(
                KnowledgeBaseItem(
                    id=int(match.get("id") or 0),
                    category=str(match.get("category") or category or "general"),
                    content=str(match.get("content") or ""),
                    timestamp=float(match.get("timestamp") or 0.0),
                    tags=list(match.get("tags") or []),
                    confidence=float(match.get("confidence") or 1.0),
                    status=str(match.get("status") or "active"),
                    metadata=dict(match.get("metadata") or {}),
                    score=None,
                )
            )
        return hits[: max(1, int(top_k))]

    def recent(
        self,
        *,
        limit: int = 20,
        category: str | None = None,
        tags: Optional[Sequence[str]] = None,
        include_inactive: bool = False,
    ) -> List[KnowledgeBaseItem]:
        """Return most recent stored memories."""

        if not self.enabled:
            return []

        items: List[KnowledgeBaseItem] = []
        try:
            rows = self._memory.get(
                category=category,
                tags=list(tags) if tags else None,
                newest_first=True,
                limit=max(0, int(limit)),
                exclude_status=None if include_inactive else ["deleted"],
                include_metadata=True,
            )
        except Exception:
            rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            items.append(
                KnowledgeBaseItem(
                    id=int(row.get("id") or 0),
                    category=str(row.get("category") or category or "general"),
                    content=str(row.get("content") or ""),
                    timestamp=float(row.get("timestamp") or 0.0),
                    tags=list(row.get("tags") or []),
                    confidence=float(row.get("confidence") or 1.0),
                    status=str(row.get("status") or "active"),
                    metadata=dict(row.get("metadata") or {}),
                    score=None,
                )
            )
        return items


__all__ = ["KnowledgeBase", "KnowledgeBaseItem"]
