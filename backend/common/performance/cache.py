"""Small multi-tier cache with optional backing store."""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Callable, Dict, Generic, Optional, Protocol, TypeVar


V = TypeVar("V")


class CacheBackend(Protocol[V]):
    def get(self, key: str) -> Optional[V]:
        ...

    def set(self, key: str, value: V, ttl_seconds: Optional[int] = None) -> None:
        ...

    def delete(self, key: str) -> None:
        ...


@dataclass
class CacheEntry(Generic[V]):
    value: V
    expires_at: float


class MultiTierCache(Generic[V]):
    def __init__(
        self,
        *,
        default_ttl_seconds: int = 60,
        backend: Optional[CacheBackend[V]] = None,
    ) -> None:
        self._default_ttl = default_ttl_seconds
        self._backend = backend
        self._lock = Lock()
        self._store: Dict[str, CacheEntry[V]] = {}

    def get(
        self,
        key: str,
        loader: Optional[Callable[[], V]] = None,
        *,
        ttl_seconds: Optional[int] = None,
    ) -> Optional[V]:
        with self._lock:
            entry = self._store.get(key)
        now = monotonic()
        if entry and entry.expires_at > now:
            return entry.value

        if entry:
            with self._lock:
                self._store.pop(key, None)

        if self._backend:
            backend_value = self._backend.get(key)
            if backend_value is not None:
                self._remember(key, backend_value, ttl_seconds or self._default_ttl)
                return backend_value

        if loader is None:
            return None

        value = loader()
        self.set(key, value, ttl_seconds=ttl_seconds)
        return value

    def set(self, key: str, value: V, *, ttl_seconds: Optional[int] = None) -> None:
        ttl = ttl_seconds or self._default_ttl
        self._remember(key, value, ttl)
        if self._backend:
            self._backend.set(key, value, ttl_seconds=ttl)

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)
        if self._backend:
            self._backend.delete(key)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def _remember(self, key: str, value: V, ttl_seconds: int) -> None:
        with self._lock:
            self._store[key] = CacheEntry(value=value, expires_at=monotonic() + ttl_seconds)
