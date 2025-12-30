from __future__ import annotations

"""Unified repositories that orchestrate caching, backends, and archival."""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from backend.autogpt.autogpt.core.knowledge_graph.graph_store import get_graph_store
from backend.autogpt.autogpt.core.knowledge_graph.ontology import EntityType, RelationType
from modules.memory import (
    VectorMemoryStore,
    build_ann_backend_from_env,
    build_archive_backend_from_env,
)
from modules.memory.vector_store import VectorRecord
from modules.memory.backends import VectorANNBackend, VectorArchiveBackend
from modules.environment import get_hardware_registry, report_resource_signal
from modules.events import EventBus

logger = logging.getLogger(__name__)


@dataclass
class HeatStats:
    reads: int = 0
    writes: int = 0
    cache_hits: int = 0
    backend_hits: int = 0
    average_latency_ms: float = 0.0
    queue_depth: int = 0


@dataclass
class KnowledgeRepository:
    """Facade coordinating graph cache/backends."""

    flush_interval: float = 0.25
    metrics: HeatStats = field(default_factory=HeatStats)

    def __post_init__(self) -> None:
        self._graph_store = get_graph_store()
        self._flush_queue: "queue.Queue[Optional[tuple[str, tuple[Any, ...], Dict[str, Any]]]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._flush_loop, name="GraphRepoFlush", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    def add_node(self, node_id: str, entity_type: EntityType, **properties: Any) -> None:
        start = time.perf_counter()
        self._graph_store.add_node(node_id, entity_type, **properties)
        self._enqueue("add_node", node_id, entity_type, properties)
        self._update_latency(time.perf_counter() - start)

    def add_edge(self, source: str, target: str, relation_type: RelationType, **properties: Any) -> None:
        start = time.perf_counter()
        self._graph_store.add_edge(source, target, relation_type, **properties)
        self._enqueue("add_edge", source, target, relation_type, properties)
        self._update_latency(time.perf_counter() - start)

    def remove_node(self, node_id: str) -> None:
        start = time.perf_counter()
        self._graph_store.remove_node(node_id)
        self._enqueue("remove_node", node_id)
        self._update_latency(time.perf_counter() - start)

    def remove_edge(self, source: str, target: str, relation_type: Optional[RelationType] = None) -> None:
        start = time.perf_counter()
        self._graph_store.remove_edge(source, target, relation_type)
        self._enqueue("remove_edge", source, target, relation_type)
        self._update_latency(time.perf_counter() - start)

    def query(
        self,
        node_id: Optional[str] = None,
        entity_type: Optional[EntityType] = None,
        relation_type: Optional[RelationType] = None,
    ) -> Dict[str, List]:
        start = time.perf_counter()
        result = self._graph_store.query(node_id=node_id, entity_type=entity_type, relation_type=relation_type)
        self._update_latency(time.perf_counter() - start, read=True)
        return result

    def stats(self) -> Dict[str, Any]:
        snapshot = {
            "reads": self.metrics.reads,
            "writes": self.metrics.writes,
            "cache_hits": self.metrics.cache_hits,
            "backend_hits": self.metrics.backend_hits,
            "avg_latency_ms": self.metrics.average_latency_ms,
            "queue_depth": self._flush_queue.qsize(),
            "version": self._graph_store.get_version(),
        }
        return snapshot

    def emit_resource_signal(
        self,
        worker_id: str,
        event_bus: Optional[EventBus] = None,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        signal = {
            "avg_latency_ms": float(self.metrics.average_latency_ms),
            "queue_depth": float(self._flush_queue.qsize()),
            "backend_hits": float(self.metrics.backend_hits),
            "reads": float(self.metrics.reads),
            "writes": float(self.metrics.writes),
        }
        meta = {"repository": "knowledge"}
        if metadata:
            meta.update(metadata)
        report_resource_signal(worker_id, signal, metadata=meta, event_bus=event_bus)

    def close(self) -> None:
        self._stop_event.set()
        self._flush_queue.put(None)
        self._worker.join(timeout=2.0)

    # ------------------------------------------------------------------
    def _enqueue(self, method: str, *args: Any) -> None:
        self._flush_queue.put((method, args, {}))
        self.metrics.writes += 1
        self.metrics.queue_depth = self._flush_queue.qsize()

    def _flush_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._flush_queue.get(timeout=self.flush_interval)
            except queue.Empty:
                continue
            if item is None:
                continue
            method, args, kwargs = item
            try:
                getattr(self._graph_store, method)(*args, **kwargs)
                self.metrics.backend_hits += 1
            except Exception:
                logger.warning("Graph repository flush failure for %s", method, exc_info=True)
            finally:
                self._flush_queue.task_done()
                self.metrics.queue_depth = self._flush_queue.qsize()
        while True:
            try:
                item = self._flush_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                continue
            method, args, kwargs = item
            try:
                getattr(self._graph_store, method)(*args, **kwargs)
            except Exception:
                logger.warning("Graph repository flush failure on shutdown for %s", method, exc_info=True)
            finally:
                self.metrics.backend_hits += 1
                self._flush_queue.task_done()

    def _update_latency(self, delta: float, *, read: bool = False) -> None:
        ms = delta * 1000.0
        weight = 0.1
        self.metrics.average_latency_ms = (
            (1 - weight) * self.metrics.average_latency_ms + weight * ms
        )
        if read:
            self.metrics.reads += 1


@dataclass
class VectorRepository:
    """Repository coordinating vector cache, ANN, and archive."""

    storage_root: Path
    embedder: EmbeddingFn | None = None
    ann_backend: Optional[VectorANNBackend] = None
    archive_backend: Optional[VectorArchiveBackend] = None
    hot_cache_size: int = 256
    metrics: HeatStats = field(default_factory=HeatStats)

    def __post_init__(self) -> None:
        ann_backend = self.ann_backend or build_ann_backend_from_env()
        archive_backend = self.archive_backend or build_archive_backend_from_env(self.storage_root)
        self._store = VectorMemoryStore(
            self.storage_root,
            embedder=self.embedder,
            ann_backend=ann_backend,
            archive_backend=archive_backend,
            hot_cache_size=self.hot_cache_size,
        )

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        start = time.perf_counter()
        record_id = self._store.add_text(text, metadata)
        self._update_latency(time.perf_counter() - start)
        return record_id

    def add_batch(self, texts: Iterable[str], metadatas: Optional[Iterable[Dict[str, Any]]] = None) -> List[str]:
        start = time.perf_counter()
        ids = self._store.add_batch(texts, metadatas)
        self._update_latency(time.perf_counter() - start)
        return ids

    def query(self, text: str, top_k: int = 5) -> List[VectorRecord]:
        start = time.perf_counter()
        results = self._store.query(text, top_k)
        self._update_latency(time.perf_counter() - start, read=True)
        return results

    def archive(self, record_id: str) -> bool:
        start = time.perf_counter()
        success = self._store.archive_record(record_id)
        self._update_latency(time.perf_counter() - start)
        return success

    def close(self) -> None:
        self._store.close()

    def stats(self) -> Dict[str, Any]:
        snapshot = {
            "records": len(self._store),
            "avg_latency_ms": self.metrics.average_latency_ms,
            "reads": self.metrics.reads,
            "writes": self.metrics.writes,
        }
        return snapshot

    def emit_resource_signal(
        self,
        worker_id: str,
        event_bus: Optional[EventBus] = None,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        signal = {
            "avg_latency_ms": float(self.metrics.average_latency_ms),
            "reads": float(self.metrics.reads),
            "writes": float(self.metrics.writes),
            "record_count": float(len(self._store)),
        }
        meta = {"repository": "vector"}
        if metadata:
            meta.update(metadata)
        report_resource_signal(worker_id, signal, metadata=meta, event_bus=event_bus)

    def _update_latency(self, delta: float, *, read: bool = False) -> None:
        ms = delta * 1000.0
        weight = 0.1
        self.metrics.average_latency_ms = (
            (1 - weight) * self.metrics.average_latency_ms + weight * ms
        )
        if read:
            self.metrics.reads += 1
        else:
            self.metrics.writes += 1


__all__ = ["KnowledgeRepository", "VectorRepository", "HeatStats"]
