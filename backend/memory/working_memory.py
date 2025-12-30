"""Short-term working memory implementation."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Iterable, Mapping, Optional


class WorkingMemory:
    """FIFO working memory with a fixed capacity."""

    def __init__(self, capacity: int = 10) -> None:
        self.capacity = capacity
        self._items: Deque[Any] = deque(maxlen=capacity)

    def store(
        self, item: Any, *, metadata: Optional[Mapping[str, Any]] = None
    ) -> None:
        """Persist ``item`` ignoring optional ``metadata``."""

        _ = metadata  # Metadata is currently unused but kept for compatibility
        self._items.append(item)

    def retrieve(
        self, filters: Optional[Mapping[str, Any]] = None
    ) -> list[Any]:
        """Return items optionally honouring ``limit``/``reverse`` filters."""

        items: Iterable[Any] = self._items
        limit: Optional[int] = None
        reverse = False
        if filters:
            limit_value = filters.get("limit")
            if isinstance(limit_value, int):
                limit = max(limit_value, 0)
            reverse = bool(filters.get("reverse", False))
        ordered = list(reversed(items)) if reverse else list(items)
        if limit is not None:
            ordered = ordered[:limit]
        return ordered

    def add(self, item: Any) -> None:
        """Backwards compatible alias for :meth:`store`."""

        self.store(item)

    def get(self) -> list[Any]:
        """Backwards compatible alias for :meth:`retrieve`."""

        return self.retrieve()

    def clear(self) -> None:
        """Remove all items from working memory."""
        self._items.clear()

    def search(
        self, query: str, *, limit: Optional[int] = None
    ) -> Iterable[Any]:
        """Return recent items that contain ``query`` as a substring."""

        results: list[Any] = []
        max_results = max(limit, 0) if isinstance(limit, int) else None
        for item in reversed(self._items):
            if max_results is not None and len(results) >= max_results:
                break
            try:
                item_text = str(item)
            except Exception:
                continue
            if query in item_text:
                results.append(item)
        return results
