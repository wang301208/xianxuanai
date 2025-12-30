"""Lightweight message bus for brain region communication.

This module defines small components for dispatching and routing neural events
between simulated brain regions.  It exposes simple ``publish_neural_event``
and ``subscribe_to_brain_region`` interfaces that perform event validation,
message routing and basic fault isolation via a circuit breaker.

The goal is clarity over sophistication so the implementation stays
straight‑forward and easily testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


# ---------------------------------------------------------------------------
# Core components
# ---------------------------------------------------------------------------


@dataclass
class EventDispatcher:
    """Store and notify event subscribers."""

    subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = field(
        default_factory=dict
    )

    def subscribe(self, region: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        if not region:
            raise ValueError("region must be non-empty")
        if not callable(handler):
            raise TypeError("handler must be callable")
        self.subscribers.setdefault(region, []).append(handler)

    def publish(self, region: str, event: Dict[str, Any]) -> None:
        for handler in self.subscribers.get(region, []):
            handler(event)


@dataclass
class CircuitBreaker:
    """Simple circuit breaker to isolate failing regions."""

    max_failures: int = 3
    failure_counts: Dict[str, int] = field(default_factory=dict)
    open_regions: set[str] = field(default_factory=set)

    def call(
        self, region: str, handler: Callable[[Dict[str, Any]], None], event: Dict[str, Any]
    ) -> None:
        if region in self.open_regions:
            return
        try:
            handler(event)
            self.failure_counts[region] = 0
        except Exception:
            count = self.failure_counts.get(region, 0) + 1
            self.failure_counts[region] = count
            if count >= self.max_failures:
                self.open_regions.add(region)


@dataclass
class MessageRouter:
    """Route events to target regions via an :class:`EventDispatcher`."""

    dispatcher: EventDispatcher

    def route(self, event: Dict[str, Any]) -> None:
        region = event.get("target")
        if not region:
            raise ValueError("event missing 'target'")
        self.dispatcher.publish(region, event)


# ---------------------------------------------------------------------------
# Module level bus instance and public API
# ---------------------------------------------------------------------------

_dispatcher = EventDispatcher()
_router = MessageRouter(_dispatcher)
_breaker = CircuitBreaker()


def reset_message_bus() -> None:
    """Reset all subscribers and breaker state (useful for tests)."""

    _dispatcher.subscribers.clear()
    _breaker.failure_counts.clear()
    _breaker.open_regions.clear()


def subscribe_to_brain_region(
    region: str, handler: Callable[[Dict[str, Any]], None]
) -> None:
    """Register a handler for events destined for ``region``."""

    def wrapped(event: Dict[str, Any]) -> None:
        _breaker.call(region, handler, event)

    _dispatcher.subscribe(region, wrapped)


def publish_neural_event(event: Dict[str, Any]) -> None:
    """Validate and publish ``event`` to its target region."""

    if not isinstance(event, dict):
        raise TypeError("event must be a dictionary")
    if "target" not in event:
        raise ValueError("event missing 'target'")
    _router.route(event)
