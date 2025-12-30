"""Utilities for managing websocket streaming to Brain API clients."""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, Optional, Set

try:  # pragma: no cover - allow running without gevent
    from gevent import spawn as _gevent_spawn
    from gevent.queue import Queue  # type: ignore
except ImportError:  # pragma: no cover - fallback for minimal environments
    _gevent_spawn = None
    from queue import Queue  # type: ignore


class StreamingHub:
    """Manage WebSocket subscribers and broadcast events."""

    def __init__(self) -> None:
        self._clients: Set[Queue] = set()

    def register_client(self) -> Queue:
        """Register a new WebSocket client and return its message queue."""
        queue: Queue = Queue()
        self._clients.add(queue)
        return queue

    def unregister_client(self, queue: Queue) -> None:
        """Remove a WebSocket client queue from the hub."""
        self._clients.discard(queue)

    def make_event(self, event: str, payload: Any) -> Dict[str, Any]:
        """Create a structured event payload for downstream clients."""
        return {
            "event": event,
            "timestamp": time.time(),
            "data": payload,
        }

    def broadcast(
        self,
        event: str,
        payload: Any,
        *,
        transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        """Broadcast an event to all registered queues."""
        if not self._clients:
            return

        data = transform(payload) if transform else payload
        message = self.make_event(event, data)

        for queue in list(self._clients):
            try:
                queue.put_nowait(message)
            except Exception:
                # Skip misbehaving client queues; they will be cleaned up on disconnect.
                continue

    def create_ws_handler(
        self,
        initial_event_factory: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
    ) -> Callable[[Any], None]:
        """Return a websocket handler that forwards hub events."""

        if _gevent_spawn is None:
            raise RuntimeError("gevent is required for websocket streaming support")

        def _handler(ws: Any) -> None:
            client_queue = self.register_client()
            reader = _gevent_spawn(self._consume_ws, ws)
            try:
                if initial_event_factory is not None:
                    initial_event = initial_event_factory()
                    if initial_event is not None:
                        try:
                            ws.send(json.dumps(initial_event))
                        except Exception:
                            pass
                while not ws.closed:
                    try:
                        message = client_queue.get()
                        if message is None:
                            break
                        ws.send(json.dumps(message))
                    except Exception:
                        break
            finally:
                try:
                    reader.kill()
                except Exception:
                    pass
                self.unregister_client(client_queue)

        return _handler

    @staticmethod
    def _consume_ws(ws: Any) -> None:
        """Continuously drain incoming messages from the socket."""
        try:
            while not ws.closed:
                ws.receive()
        except Exception:
            pass


__all__ = ["StreamingHub"]
