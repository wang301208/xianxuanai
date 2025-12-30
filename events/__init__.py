"""Compatibility wrappers for the optional :mod:`modules.events` package."""

from __future__ import annotations

from modules.events import (  # type: ignore[import-not-found]
    AsyncEventBus,
    EventBus,
    InMemoryEventBus,
    create_event_bus,
    publish,
    subscribe,
    unsubscribe,
)

__all__ = [
    "EventBus",
    "InMemoryEventBus",
    "AsyncEventBus",
    "create_event_bus",
    "publish",
    "subscribe",
    "unsubscribe",
]
