"""Least Frequently Used (LFU) cache implementation."""
from collections import defaultdict, OrderedDict
from typing import Any

from ...base import Algorithm


class LFUCache(Algorithm):
    """LFU cache that evicts the least frequently used items first.

    Uses frequency buckets to provide ``O(1)`` ``get`` and ``put`` operations.
    When multiple keys have the same frequency, the least recently used one is
    discarded.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.cache: dict[Any, Any] = {}
        self.freq: dict[Any, int] = defaultdict(int)
        self.buckets: dict[int, "OrderedDict[Any, Any]"] = defaultdict(OrderedDict)
        self.min_freq = 0

    def _update(self, key: Any, value: Any | None = None) -> None:
        freq = self.freq[key]
        val = self.buckets[freq].pop(key)
        if not self.buckets[freq]:
            del self.buckets[freq]
            if self.min_freq == freq:
                self.min_freq += 1
        self.freq[key] += 1
        new_freq = self.freq[key]
        self.buckets[new_freq][key] = value if value is not None else val

    def get(self, key: Any) -> Any | None:
        """Return value for ``key`` or ``None`` if missing."""
        if key not in self.cache:
            return None
        self._update(key)
        return self.cache[key]

    def put(self, key: Any, value: Any) -> None:
        """Insert or update a value.

        Evicts the least frequently used item when the cache is full.
        """
        if self.capacity <= 0:
            return
        if key in self.cache:
            self.cache[key] = value
            self._update(key, value)
            return
        if len(self.cache) >= self.capacity:
            k, v = self.buckets[self.min_freq].popitem(last=False)
            del self.cache[k]
            del self.freq[k]
            if not self.buckets[self.min_freq]:
                del self.buckets[self.min_freq]
        self.cache[key] = value
        self.freq[key] = 1
        self.buckets[1][key] = value
        self.min_freq = 1

    def execute(self, *args, **kwargs) -> dict[Any, Any]:
        """Return a snapshot of the current cache state."""
        return dict(self.cache)
