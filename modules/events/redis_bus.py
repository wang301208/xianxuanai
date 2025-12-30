"""Redis based event bus implementation."""

from __future__ import annotations

import json
import threading
from typing import Any, Callable, Dict, Tuple

import redis

from third_party.autogpt.autogpt.core.errors import AutoGPTError
from third_party.autogpt.autogpt.core.logging import handle_exception
from . import EventBus


class RedisEventBus(EventBus):
    """Event bus using Redis Pub/Sub."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: str | None = None,
        db: int = 0,
    ) -> None:
        self._redis = redis.Redis(host=host, port=port, password=password, db=db)
        self._subscriptions: Dict[Tuple[str, Callable[[Dict[str, Any]], None]], Callable[[], None]] = {}

    def publish(self, topic: str, event: Dict[str, Any]) -> None:
        self._redis.publish(topic, json.dumps(event))

    def subscribe(
        self, topic: str, handler: Callable[[Dict[str, Any]], None]
    ) -> Callable[[], None]:
        pubsub = self._redis.pubsub()
        pubsub.subscribe(topic)

        def _listen() -> None:
            for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                try:
                    data = json.loads(message["data"])
                except json.JSONDecodeError as err:
                    handle_exception(err)
                    continue
                try:
                    handler(data)
                except AutoGPTError as err:
                    handle_exception(err)

        thread = threading.Thread(target=_listen, daemon=True)
        thread.start()

        def cancel() -> None:
            pubsub.close()
            thread.join(timeout=1)

        self._subscriptions[(topic, handler)] = cancel
        return cancel

    def unsubscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        cancel = self._subscriptions.pop((topic, handler), None)
        if cancel:
            cancel()


__all__ = ["RedisEventBus"]

