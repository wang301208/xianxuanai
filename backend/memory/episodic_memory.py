"""Episodic memory storing time stamped experiences."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Mapping, Optional


@dataclass
class Episode:
    timestamp: float
    data: str


class EpisodicMemory:
    """Simple list based episodic memory."""

    def __init__(self) -> None:
        self._episodes: List[Episode] = []

    def store(
        self, item: str, *, metadata: Optional[Mapping[str, float]] = None
    ) -> None:
        """Record a new episode using the unified protocol."""

        timestamp = None
        if metadata is not None:
            timestamp = metadata.get("timestamp")
        if timestamp is None:
            timestamp = datetime.utcnow().timestamp()
        self._episodes.append(Episode(float(timestamp), item))

    def retrieve(
        self, filters: Optional[Mapping[str, float]] = None
    ) -> Iterable[str]:
        """Yield episodes filtered by optional ``start_ts``/``end_ts``."""

        start_ts = filters.get("start_ts") if filters else None
        end_ts = filters.get("end_ts") if filters else None
        for ep in self._episodes:
            if start_ts is not None and ep.timestamp < start_ts:
                continue
            if end_ts is not None and ep.timestamp >= end_ts:
                continue
            yield ep.data

    def add(self, data: str, *, timestamp: Optional[float] = None) -> None:
        """Backwards compatible alias for :meth:`store`."""

        metadata = {"timestamp": timestamp} if timestamp is not None else None
        self.store(data, metadata=metadata)

    def get(
        self,
        *,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
    ) -> Iterable[str]:
        """Backwards compatible alias for :meth:`retrieve`."""

        filters = {}
        if start_ts is not None:
            filters["start_ts"] = start_ts
        if end_ts is not None:
            filters["end_ts"] = end_ts
        return self.retrieve(filters)

    def clear(self) -> None:
        """Remove all stored episodes."""
        self._episodes.clear()

    def search(
        self, query: str, *, limit: Optional[int] = None
    ) -> Iterable[str]:  # pragma: no cover - not used in tests
        """Return episodes whose payload contains ``query``."""

        matched: list[str] = []
        for ep in self._episodes:
            if query.lower() in ep.data.lower():
                matched.append(ep.data)
                if limit is not None and len(matched) >= limit:
                    break
        return matched
