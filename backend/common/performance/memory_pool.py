"""Reusable object pool to reduce hot-path allocations."""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from threading import Lock
from typing import Callable, Deque, Generator, Generic, TypeVar


T = TypeVar("T")


class ObjectPool(Generic[T]):
    def __init__(self, factory: Callable[[], T], *, max_size: int = 32) -> None:
        self._factory = factory
        self._max_size = max_size
        self._lock = Lock()
        self._pool: Deque[T] = deque()
        self._created = 0

    def acquire(self) -> T:
        with self._lock:
            try:
                return self._pool.popleft()
            except IndexError:
                self._created += 1
        return self._factory()

    def release(self, item: T) -> None:
        with self._lock:
            if len(self._pool) < self._max_size:
                self._pool.append(item)

    @contextmanager
    def borrow(self) -> Generator[T, None, None]:
        resource = self.acquire()
        try:
            yield resource
        finally:
            self.release(resource)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "capacity": self._max_size,
                "available": len(self._pool),
                "created": self._created,
            }
