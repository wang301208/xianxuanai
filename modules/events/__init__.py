"""Event bus abstraction for AutoGPT.

This module defines a minimal publish/subscribe interface and exposes helper
functions for interacting with a global event bus instance. The default
implementation is an in-memory bus, but alternative backends such as Redis can
be plugged in to enable coordination across multiple hosts.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, List

from third_party.autogpt.autogpt.core.errors import AutoGPTError
from third_party.autogpt.autogpt.core.logging import handle_exception

__all__ = [
    "EventBus",
    "InMemoryEventBus",
    "AsyncEventBus",
    "create_event_bus",
    "publish",
    "subscribe",
    "unsubscribe",
]


class EventBus(ABC):
    """Simple publish/subscribe interface used by AutoGPT."""

    @abstractmethod
    def publish(self, topic: str, event: Dict[str, Any]) -> None:
        """Publish *event* on *topic*."""

    @abstractmethod
    def subscribe(
        self, topic: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> Callable[[], None]:
        """Subscribe *handler* to events on *topic* and return a cancel function."""

    @abstractmethod
    def unsubscribe(
        self, topic: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> None:
        """Remove *handler* subscription from *topic*."""


class InMemoryEventBus(EventBus):
    """Simple in-memory pub/sub event bus using worker threads."""

    def __init__(self, worker_count: int = 1) -> None:
        self._subscribers: Dict[
            str, List[Callable[[Dict[str, Any]], Awaitable[None]]]
        ] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[tuple[str, Dict[str, Any]]] = queue.Queue()
        self._workers: List[threading.Thread] = []
        for _ in range(max(1, worker_count)):
            worker = threading.Thread(target=self._worker, daemon=True)
            worker.start()
            self._workers.append(worker)

    def _worker(self) -> None:
        while True:
            topic, event = self._queue.get()
            try:
                with self._lock:
                    subscribers = list(self._subscribers.get(topic, []))
                for handler in subscribers:
                    try:
                        result = handler(event)
                        if asyncio.iscoroutine(result):
                            asyncio.run(result)
                    except AutoGPTError as err:
                        handle_exception(err)
            finally:
                self._queue.task_done()

    def publish(self, topic: str, event: Dict[str, Any]) -> None:
        """Enqueue *event* to be processed asynchronously."""

        self._queue.put((topic, event))

    def subscribe(
        self, topic: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> Callable[[], None]:
        with self._lock:
            self._subscribers.setdefault(topic, []).append(handler)
        return lambda: self.unsubscribe(topic, handler)

    def unsubscribe(
        self, topic: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> None:
        with self._lock:
            handlers = self._subscribers.get(topic)
            if handlers and handler in handlers:
                handlers.remove(handler)
                if not handlers:
                    del self._subscribers[topic]

    def join(self) -> None:
        """Block until all queued events have been processed."""

        self._queue.join()


class AsyncEventBus(EventBus):
    """Event bus based on ``asyncio`` for coroutine handlers."""

    def __init__(self) -> None:
        self._subscribers: Dict[
            str, List[Callable[[Dict[str, Any]], Awaitable[None]]]
        ] = {}
        self._lock = asyncio.Lock()
        self._queue: asyncio.Queue[tuple[str, Dict[str, Any]]] = asyncio.Queue()
        self._worker_task = asyncio.create_task(self._worker())

    async def _worker(self) -> None:
        while True:
            topic, event = await self._queue.get()
            try:
                async with self._lock:
                    subscribers = list(self._subscribers.get(topic, []))
                for handler in subscribers:
                    asyncio.create_task(self._dispatch(handler, event))
            finally:
                self._queue.task_done()

    async def _dispatch(
        self, handler: Callable[[Dict[str, Any]], Awaitable[None]], event: Dict[str, Any]
    ) -> None:
        try:
            await handler(event)
        except AutoGPTError as err:  # pragma: no cover - logging only
            handle_exception(err)

    def publish(self, topic: str, event: Dict[str, Any]) -> None:
        self._queue.put_nowait((topic, event))

    def subscribe(
        self, topic: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> Callable[[], None]:
        async def _subscribe() -> None:
            async with self._lock:
                self._subscribers.setdefault(topic, []).append(handler)

        asyncio.create_task(_subscribe())
        return lambda: self.unsubscribe(topic, handler)

    def unsubscribe(
        self, topic: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> None:
        async def _unsubscribe() -> None:
            async with self._lock:
                handlers = self._subscribers.get(topic)
                if handlers and handler in handlers:
                    handlers.remove(handler)
                    if not handlers:
                        del self._subscribers[topic]

        asyncio.create_task(_unsubscribe())

    async def join(self) -> None:
        await self._queue.join()


def create_event_bus(backend: str | None = None, **kwargs: Any) -> EventBus:
    """Create an event bus for *backend*.

    ``backend`` may be ``"async"`` (default) for :class:`AsyncEventBus`,
    ``"inmemory"``/``"thread"`` for :class:`InMemoryEventBus`, or ``"redis"`` for
    :class:`RedisEventBus`.
    """

    backend = (backend or "async").lower()
    if backend == "redis":
        from .redis_bus import RedisEventBus

        return RedisEventBus(**kwargs)
    if backend in {"thread", "inmemory"}:
        return InMemoryEventBus()
    return AsyncEventBus()


def publish(bus: EventBus, topic: str, event: Dict[str, Any]) -> None:
    """Publish *event* on *topic* using *bus*."""

    bus.publish(topic, event)


def subscribe(
    bus: EventBus, topic: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]
) -> Callable[[], None]:
    """Subscribe *handler* to events published on *topic* using *bus*."""

    return bus.subscribe(topic, handler)


def unsubscribe(
    bus: EventBus, topic: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]
) -> None:
    """Remove *handler* subscription from *topic* using *bus*."""

    bus.unsubscribe(topic, handler)

