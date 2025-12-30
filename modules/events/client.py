"""Lightweight client for interacting with the AutoGPT event bus."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict

from . import EventBus


class EventClient:
    """Convenience wrapper for publishing and subscribing to events.

    Other services can use this client without needing to know about the underlying
    event bus implementation.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    def publish(self, topic: str, event: Dict[str, Any]) -> None:
        self._bus.publish(topic, event)

    def subscribe(
        self, topic: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> Callable[[], None]:
        return self._bus.subscribe(topic, handler)

    def unsubscribe(
        self, topic: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> None:
        self._bus.unsubscribe(topic, handler)
