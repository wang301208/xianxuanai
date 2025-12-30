from __future__ import annotations

"""Shard-aware vector memory helpers with optional Ray coordination."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .vector_store import VectorMemoryStore, VectorRecord

logger = logging.getLogger(__name__)

try:  # optional dependency
    import ray  # type: ignore

    _HAS_RAY = True
except Exception:  # pragma: no cover - optional dependency
    ray = None  # type: ignore
    _HAS_RAY = False


def _record_to_dict(record: VectorRecord) -> Dict[str, Any]:
    return {
        "id": record.id,
        "text": record.text,
        "metadata": dict(record.metadata),
        "score": record.score,
    }


class VectorShardCoordinator:
    """Coordinate vector upserts/queries while tracking shard metadata."""

    def __init__(self, storage_path: str | Path, **store_kwargs: Any) -> None:
        self._store = VectorMemoryStore(Path(storage_path), **store_kwargs)
        self._logger = logger

    # ------------------------------------------------------------------ write path
    def upsert(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        record_id: Optional[str] = None,
    ) -> str:
        return self._store.add_text(text, metadata, record_id=record_id)

    def batch_upsert(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[str]:
        return self._store.add_batch(texts, metadatas)

    # ------------------------------------------------------------------ read path
    def query(
        self,
        text: str,
        top_k: int = 5,
        *,
        shard: Optional[str] = None,
        shards: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        records = self._store.query(text, top_k=top_k, shard=shard, shards=shards)
        return [_record_to_dict(record) for record in records]

    # ------------------------------------------------------------------ maintenance
    def archive(self, record_id: str) -> bool:
        return self._store.archive_record(record_id)

    def restore(self, record_id: str) -> bool:
        return self._store.restore_record(record_id)

    def maintain_heat(
        self,
        *,
        max_hot: Optional[int] = None,
        idle_seconds: Optional[float] = None,
        batch_size: int = 64,
    ) -> Dict[str, int]:
        return self._store.run_heat_maintenance(
            max_hot=max_hot, idle_seconds=idle_seconds, batch_size=batch_size
        )

    def shards(self) -> Dict[str, float]:
        return self._store.get_shard_stats()

    def record_access(self, record_id: str) -> Optional[Dict[str, Any]]:
        return self._store.record_access(record_id)

    def close(self) -> None:
        self._store.close()


class _RayVectorCoordinatorHandle:
    """Synchronous wrapper around the Ray actor implementation."""

    def __init__(self, actor_handle: "ray.actor.ActorHandle") -> None:  # type: ignore[name-defined]
        self._actor = actor_handle

    def upsert(self, text: str, metadata: Optional[Dict[str, Any]] = None, record_id: Optional[str] = None) -> str:
        return ray.get(self._actor.upsert.remote(text, metadata, record_id))  # type: ignore[attr-defined]

    def batch_upsert(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[str]:
        return list(ray.get(self._actor.batch_upsert.remote(texts, metadatas)))  # type: ignore[attr-defined]

    def query(
        self,
        text: str,
        top_k: int = 5,
        *,
        shard: Optional[str] = None,
        shards: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        return list(
            ray.get(self._actor.query.remote(text, top_k, shard=shard, shards=shards))  # type: ignore[attr-defined]
        )

    def archive(self, record_id: str) -> bool:
        return bool(ray.get(self._actor.archive.remote(record_id)))  # type: ignore[attr-defined]

    def restore(self, record_id: str) -> bool:
        return bool(ray.get(self._actor.restore.remote(record_id)))  # type: ignore[attr-defined]

    def maintain_heat(
        self,
        *,
        max_hot: Optional[int] = None,
        idle_seconds: Optional[float] = None,
        batch_size: int = 64,
    ) -> Dict[str, int]:
        return dict(
            ray.get(
                self._actor.maintain_heat.remote(  # type: ignore[attr-defined]
                    max_hot=max_hot, idle_seconds=idle_seconds, batch_size=batch_size
                )
            )
        )

    def shards(self) -> Dict[str, float]:
        return dict(ray.get(self._actor.shards.remote()))  # type: ignore[attr-defined]

    def record_access(self, record_id: str) -> Optional[Dict[str, Any]]:
        return ray.get(self._actor.record_access.remote(record_id))  # type: ignore[attr-defined]

    def close(self) -> None:
        ray.get(self._actor.close.remote())  # type: ignore[attr-defined]


def create_ray_vector_coordinator(storage_path: str | Path, **store_kwargs: Any) -> _RayVectorCoordinatorHandle:
    """Create a Ray actor that proxies a :class:`VectorShardCoordinator`."""

    if not _HAS_RAY:
        raise RuntimeError("Ray is not available; install ray to use the vector coordinator actor.")
    actor_cls = ray.remote(VectorShardCoordinator)  # type: ignore[misc]
    handle = actor_cls.remote(str(storage_path), **store_kwargs)  # type: ignore[attr-defined]
    return _RayVectorCoordinatorHandle(handle)


__all__ = ["VectorShardCoordinator", "create_ray_vector_coordinator"]
