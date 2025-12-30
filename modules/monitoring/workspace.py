from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class WorkspaceMessage:
    type: str
    source: str
    payload: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[str] = None
    tags: Tuple[str, ...] = ()
    importance: float = 0.0
    timestamp: float = field(default_factory=time.time)
    cursor: int = 0


class GlobalWorkspace:
    """In-memory pub/sub workspace used for lightweight agent telemetry."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._messages: List[WorkspaceMessage] = []
        self._cursor = 0
        self._state: Dict[str, Any] = {}

    def publish_message(
        self,
        message: WorkspaceMessage,
        *,
        attention: Optional[Sequence[float] | float] = None,
        propagate: bool = True,
    ) -> WorkspaceMessage:
        del attention, propagate
        with self._lock:
            self._cursor += 1
            stamped = WorkspaceMessage(
                type=message.type,
                source=message.source,
                payload=dict(message.payload or {}),
                summary=message.summary,
                tags=tuple(message.tags or ()),
                importance=float(message.importance or 0.0),
                timestamp=float(message.timestamp or time.time()),
                cursor=self._cursor,
            )
            self._messages.append(stamped)
            return stamped

    def get_updates(
        self,
        *,
        cursor: Optional[int] = None,
        limit: int = 10,
        types: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[str]] = None,
        exclude_sources: Optional[Sequence[str]] = None,
    ) -> Tuple[List[WorkspaceMessage], Optional[int]]:
        with self._lock:
            start = int(cursor or 0)
            items: Iterable[WorkspaceMessage] = (m for m in self._messages if m.cursor > start)
            if exclude_sources:
                excluded = set(exclude_sources)
                items = (m for m in items if m.source not in excluded)
            if types:
                allowed = set(types)
                items = (m for m in items if m.type in allowed)
            if tags:
                required = set(tags)
                items = (m for m in items if required.intersection(m.tags))

            selected = list(items)
            selected.sort(key=lambda m: m.cursor)
            selected = selected[: max(0, int(limit))]
            new_cursor = selected[-1].cursor if selected else cursor
            return selected, new_cursor

    def state(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._state.get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        with self._lock:
            self._state[key] = value


global_workspace = GlobalWorkspace()

