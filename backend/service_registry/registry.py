"""Simple service registry with in-memory implementation and TTL heartbeats."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Dict, List, Optional, Protocol
from uuid import uuid4


@dataclass(slots=True)
class ServiceInfo:
    name: str
    host: str
    port: int
    instance_id: str = field(default_factory=lambda: str(uuid4()))
    tags: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None

    def endpoint(self) -> str:
        scheme = self.metadata.get("scheme", "http")
        return f"{scheme}://{self.host}:{self.port}"


class ServiceRegistry(Protocol):
    def register(self, info: ServiceInfo, *, ttl_seconds: Optional[int] = None) -> ServiceInfo:
        ...

    def heartbeat(self, name: str, instance_id: str, *, ttl_seconds: Optional[int] = None) -> None:
        ...

    def deregister(self, name: str, instance_id: str) -> None:
        ...

    def list(self, name: Optional[str] = None) -> List[ServiceInfo]:
        ...


class InMemoryServiceRegistry(ServiceRegistry):
    """Thread-safe in-memory registry suitable for tests and local dev."""

    def __init__(self) -> None:
        self._services: Dict[str, Dict[str, ServiceInfo]] = {}
        self._lock = RLock()

    def register(self, info: ServiceInfo, *, ttl_seconds: Optional[int] = None) -> ServiceInfo:
        entry = self._with_expiry(info, ttl_seconds)
        with self._lock:
            bucket = self._services.setdefault(info.name, {})
            bucket[entry.instance_id] = entry
        return entry

    def heartbeat(self, name: str, instance_id: str, *, ttl_seconds: Optional[int] = None) -> None:
        if ttl_seconds is None:
            return
        with self._lock:
            bucket = self._services.get(name)
            if not bucket or instance_id not in bucket:
                raise KeyError(f"Service {name}/{instance_id} not registered")
            bucket[instance_id] = self._with_expiry(bucket[instance_id], ttl_seconds)

    def deregister(self, name: str, instance_id: str) -> None:
        with self._lock:
            bucket = self._services.get(name)
            if not bucket:
                return
            bucket.pop(instance_id, None)
            if not bucket:
                self._services.pop(name, None)

    def list(self, name: Optional[str] = None) -> List[ServiceInfo]:
        with self._lock:
            self._purge_expired()
            if name is None:
                return [entry for bucket in self._services.values() for entry in bucket.values()]
            bucket = self._services.get(name, {})
            return list(bucket.values())

    def _purge_expired(self) -> None:
        now = datetime.now(timezone.utc)
        for name, bucket in list(self._services.items()):
            for instance_id, info in list(bucket.items()):
                if info.expires_at and info.expires_at <= now:
                    bucket.pop(instance_id, None)
            if not bucket:
                self._services.pop(name, None)

    @staticmethod
    def _with_expiry(info: ServiceInfo, ttl_seconds: Optional[int]) -> ServiceInfo:
        if ttl_seconds:
            return ServiceInfo(
                name=info.name,
                host=info.host,
                port=info.port,
                instance_id=info.instance_id,
                tags=tuple(info.tags),
                metadata=dict(info.metadata),
                registered_at=datetime.now(timezone.utc),
                expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
            )
        return ServiceInfo(
            name=info.name,
            host=info.host,
            port=info.port,
            instance_id=info.instance_id,
            tags=tuple(info.tags),
            metadata=dict(info.metadata),
            registered_at=datetime.now(timezone.utc),
            expires_at=None,
        )
