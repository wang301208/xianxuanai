from __future__ import annotations

"""Collect agent lifecycle and resource metrics into SQLite."""

from pathlib import Path
from typing import Callable, List

from events import EventBus

from .storage import TimeSeriesStorage


class MetricsCollector:
    """Subscribe to agent events on the event bus and persist them."""

    def __init__(self, bus: EventBus, db_path: Path | str = "monitoring.db") -> None:
        self.storage = TimeSeriesStorage(db_path)
        self._bus = bus
        self._subscriptions: List[Callable[[], None]] = []
        self._subscriptions.append(
            self._bus.subscribe("agent.lifecycle", self._store_lifecycle)
        )
        self._subscriptions.append(
            self._bus.subscribe("agent.resource", self._store_resource)
        )

    # ------------------------------------------------------------------
    async def _store_lifecycle(self, event: dict) -> None:
        self.storage.store("agent.lifecycle", event)

    async def _store_resource(self, event: dict) -> None:
        self.storage.store("agent.resource", event)

    # ------------------------------------------------------------------
    def close(self) -> None:
        for cancel in self._subscriptions:
            cancel()
        self.storage.close()
