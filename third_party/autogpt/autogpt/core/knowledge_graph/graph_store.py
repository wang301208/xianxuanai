"""A cached graph-store facade with optional distributed backends."""

from __future__ import annotations

import atexit
import logging
import os
import queue
import threading
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .ontology import EntityType, RelationType
from .backends import GraphBackend, build_backend_from_env


logger = logging.getLogger(__name__)


@dataclass
class Node:
    id: str
    type: EntityType
    properties: Dict[str, object] = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    type: RelationType
    properties: Dict[str, object] = field(default_factory=dict)


class GraphStore:
    """In-memory graph cache with optional write-through backend."""

    def __init__(
        self,
        backend: Optional[GraphBackend] = None,
        *,
        async_flush: bool = True,
        flush_interval: float = 0.25,
    ) -> None:
        self._nodes: Dict[str, Node] = {}
        self._edges: List[Edge] = []
        self._edge_index: set[Tuple[str, str, RelationType]] = set()
        self._version: int = 0
        self._snapshots: Dict[int, Tuple[Dict[str, Node], List[Edge]]] = {}

        self._lock = threading.RLock()
        self._backend = backend
        self._async_flush = async_flush
        self._flush_interval = flush_interval
        self._write_queue: "queue.Queue[Optional[Tuple[str, Tuple[Any, ...], Dict[str, Any]]]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._closed = False
        self._wal: deque[Tuple[str, Tuple[Any, ...], Dict[str, Any], float]] = deque()
        self._wal_lock = threading.Lock()
        self._replay_lock = threading.Lock()
        self._backend_max_retries = max(1, int(os.getenv("GRAPH_BACKEND_MAX_RETRIES", "3") or 3))
        self._backend_retry_backoff = float(os.getenv("GRAPH_BACKEND_RETRY_BACKOFF", "0.5") or 0.5)
        self._backend_retry_backoff_max = float(
            os.getenv("GRAPH_BACKEND_RETRY_BACKOFF_MAX", "5.0") or 5.0
        )

        if self._backend is not None and self._async_flush:
            self._worker = threading.Thread(
                target=self._flush_loop, name="GraphStoreFlush", daemon=True
            )
            self._worker.start()

        with self._lock:
            self._cache_version_locked()

    # ------------------------------------------------------------------
    # Public API
    def add_node(
        self,
        node_id: str,
        entity_type: EntityType,
        **properties: object,
    ) -> None:
        with self._lock:
            self._nodes[node_id] = Node(node_id, entity_type, dict(properties))
            self._bump_version_locked()
        self._queue_backend_call(
            "add_node",
            node_id,
            entity_type.value if isinstance(entity_type, EntityType) else str(entity_type),
            dict(properties),
        )

    def add_edge(
        self,
        source: str,
        target: str,
        relation_type: RelationType,
        **properties: object,
    ) -> None:
        key = (source, target, relation_type)
        with self._lock:
            if key in self._edge_index:
                for edge in self._edges:
                    if edge.source == source and edge.target == target and edge.type == relation_type:
                        edge.properties.update(properties)
                        break
            else:
                edge = Edge(source, target, relation_type, dict(properties))
                self._edges.append(edge)
                self._edge_index.add(key)
            self._bump_version_locked()
        self._queue_backend_call(
            "add_edge",
            source,
            target,
            relation_type.value if isinstance(relation_type, RelationType) else str(relation_type),
            dict(properties),
        )

    def remove_node(self, node_id: str) -> None:
        with self._lock:
            if node_id in self._nodes:
                self._nodes.pop(node_id, None)
                self._edges = [
                    edge for edge in self._edges if edge.source != node_id and edge.target != node_id
                ]
                self._edge_index = {
                    (edge.source, edge.target, edge.type) for edge in self._edges
                }
                self._bump_version_locked()
        self._queue_backend_call("remove_node", node_id)

    def remove_edge(
        self,
        source: str,
        target: str,
        relation_type: RelationType | None = None,
    ) -> None:
        with self._lock:
            self._edges = [
                edge
                for edge in self._edges
                if not (
                    edge.source == source
                    and edge.target == target
                    and (relation_type is None or edge.type == relation_type)
                )
            ]
            self._edge_index = {
                (edge.source, edge.target, edge.type) for edge in self._edges
            }
            self._bump_version_locked()
        relation_value = (
            relation_type.value if isinstance(relation_type, RelationType) else relation_type
        )
        self._queue_backend_call("remove_edge", source, target, relation_value)

    def query(
        self,
        node_id: Optional[str] = None,
        entity_type: Optional[EntityType] = None,
        relation_type: Optional[RelationType] = None,
    ) -> Dict[str, List]:
        with self._lock:
            nodes = [
                node
                for node in self._nodes.values()
                if (not entity_type or node.type == entity_type)
                and (not node_id or node.id == node_id)
            ]
            edges = [
                edge
                for edge in self._edges
                if (not relation_type or edge.type == relation_type)
                and (
                    not node_id
                    or edge.source == node_id
                    or edge.target == node_id
                )
            ]

        should_fallback = (
            self._backend is not None
            and (
                (node_id and not nodes)
                or (entity_type and not nodes)
                or (not node_id and not entity_type and not self._nodes)
            )
        )

        if should_fallback:
            backend_entity_type = (
                entity_type.value if isinstance(entity_type, EntityType) else entity_type
            )
            backend_relation_type = (
                relation_type.value if isinstance(relation_type, RelationType) else relation_type
            )
            try:
                result = self._backend.query(
                    node_id=node_id,
                    entity_type=backend_entity_type,
                    relation_type=backend_relation_type,
                )
            except Exception:
                logger.debug("Graph backend query failed.", exc_info=True)
            else:
                if result:
                    self._hydrate_from_backend(result)
                with self._lock:
                    nodes = [
                        node
                        for node in self._nodes.values()
                        if (not entity_type or node.type == entity_type)
                        and (not node_id or node.id == node_id)
                    ]
                    edges = [
                        edge
                        for edge in self._edges
                        if (not relation_type or edge.type == relation_type)
                        and (
                            not node_id
                            or edge.source == node_id
                            or edge.target == node_id
                        )
                    ]

        return {"nodes": nodes, "edges": edges}

    def get_version(self) -> int:
        with self._lock:
            return self._version

    def get_snapshot(self, version: int | None = None) -> Dict[str, List]:
        with self._lock:
            version = self._version if version is None else version
            nodes, edges = self._snapshots.get(version, (self._nodes, self._edges))
            return {"nodes": list(nodes.values()), "edges": list(edges)}

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if self._worker is not None:
            self._stop_event.set()
            self._write_queue.put(None)
            self._worker.join(timeout=2.0)
        while True:
            try:
                item = self._write_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                continue
            method, args, kwargs = item
            self._safe_invoke_backend(method, args, kwargs)
            self._write_queue.task_done()
        if self._backend is not None:
            self._replay_wal()
            remaining = self._pending_wal_items()
            if remaining:
                logger.warning(
                    "Graph backend still unavailable; %s WAL operations pending on shutdown.",
                    remaining,
                )
            try:
                self._backend.close()
            except Exception:
                logger.debug("Error closing graph backend.", exc_info=True)

    # ------------------------------------------------------------------
    # Internal helpers
    def _hydrate_from_backend(self, result: Dict[str, Any]) -> None:
        changed = False
        backend_nodes = result.get("nodes") or []
        backend_edges = result.get("edges") or []
        with self._lock:
            for raw in backend_nodes:
                node_id = raw.get("id")
                node_type = raw.get("type")
                if not node_id or not node_type:
                    continue
                try:
                    entity = EntityType(node_type)
                except ValueError:
                    logger.debug("Unknown entity type from backend: %s", node_type)
                    continue
                properties = dict(raw.get("properties") or {})
                if node_id not in self._nodes:
                    changed = True
                self._nodes[node_id] = Node(node_id, entity, properties)
            for raw in backend_edges:
                source = raw.get("source")
                target = raw.get("target")
                rel_type = raw.get("type")
                if not source or not target or not rel_type:
                    continue
                try:
                    relation = RelationType(rel_type)
                except ValueError:
                    logger.debug("Unknown relation type from backend: %s", rel_type)
                    continue
                properties = dict(raw.get("properties") or {})
                key = (source, target, relation)
                if key in self._edge_index:
                    for edge in self._edges:
                        if edge.source == source and edge.target == target and edge.type == relation:
                            edge.properties.update(properties)
                            break
                else:
                    self._edges.append(Edge(source, target, relation, properties))
                    self._edge_index.add(key)
                    changed = True
            if changed:
                self._bump_version_locked()

    def _queue_backend_call(self, method: str, *args: Any, **kwargs: Any) -> None:
        if self._backend is None:
            return
        item = (method, args, kwargs)
        if self._async_flush and self._worker is not None:
            self._write_queue.put(item)
        else:
            self._safe_invoke_backend(method, args, kwargs)

    def _flush_loop(self) -> None:  # pragma: no cover - background thread
        while not self._stop_event.is_set():
            try:
                item = self._write_queue.get(timeout=self._flush_interval)
            except queue.Empty:
                continue
            if item is None:
                continue
            method, args, kwargs = item
            self._safe_invoke_backend(method, args, kwargs)
            self._write_queue.task_done()
        while True:
            try:
                item = self._write_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                continue
            method, args, kwargs = item
            self._safe_invoke_backend(method, args, kwargs)
            self._write_queue.task_done()

    def _safe_invoke_backend(self, method: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        if self._backend is None:
            return
        self._replay_wal()
        success = self._attempt_backend_call(method, args, kwargs)
        if not success:
            self._append_wal(method, args, kwargs)

    def _attempt_backend_call(
        self, method: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> bool:
        if self._backend is None:
            return True
        func = getattr(self._backend, method, None)
        if func is None:
            logger.debug("Backend missing method %s", method)
            return True
        attempt = 0
        while attempt < self._backend_max_retries:
            try:
                func(*args, **kwargs)
                return True
            except Exception:
                attempt += 1
                logger.warning(
                    "Graph backend method %s failed (attempt %s/%s).",
                    method,
                    attempt,
                    self._backend_max_retries,
                    exc_info=True,
                )
                if hasattr(self._backend, "reconnect"):
                    try:
                        self._backend.reconnect()
                    except Exception:
                        logger.debug("Graph backend reconnect attempt failed.", exc_info=True)
                if attempt < self._backend_max_retries:
                    delay = min(
                        self._backend_retry_backoff * (2 ** (attempt - 1)),
                        self._backend_retry_backoff_max,
                    )
                    time.sleep(delay)
        return False

    def _append_wal(self, method: str, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        with self._wal_lock:
            self._wal.append((method, args, kwargs, time.time()))

    def _pending_wal_items(self) -> int:
        with self._wal_lock:
            return len(self._wal)

    def _replay_wal(self) -> None:
        if self._backend is None:
            return
        if not self._wal:
            return
        if not self._replay_lock.acquire(blocking=False):
            return
        try:
            while True:
                with self._wal_lock:
                    if not self._wal:
                        break
                    method, args, kwargs, _ = self._wal.popleft()
                success = self._attempt_backend_call(method, args, kwargs)
                if not success:
                    with self._wal_lock:
                        self._wal.appendleft((method, args, kwargs, time.time()))
                    break
        finally:
            self._replay_lock.release()

    def _bump_version_locked(self) -> None:
        self._version += 1
        self._cache_version_locked()

    def _cache_version_locked(self) -> None:
        self._snapshots[self._version] = (
            deepcopy(self._nodes),
            deepcopy(self._edges),
        )

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass


_GLOBAL_GRAPH_STORE: GraphStore | None = None


def _close_global_graph_store() -> None:
    global _GLOBAL_GRAPH_STORE
    if _GLOBAL_GRAPH_STORE is not None:
        try:
            _GLOBAL_GRAPH_STORE.close()
        finally:
            _GLOBAL_GRAPH_STORE = None


def get_graph_store() -> GraphStore:
    """Return the module-wide graph store instance."""

    global _GLOBAL_GRAPH_STORE
    if _GLOBAL_GRAPH_STORE is None:
        backend = build_backend_from_env()
        _GLOBAL_GRAPH_STORE = GraphStore(backend=backend)
        atexit.register(_close_global_graph_store)
    return _GLOBAL_GRAPH_STORE


def query_graph(**kwargs: Any) -> Dict[str, List]:
    """Convenience wrapper around :meth:`GraphStore.query`."""

    return get_graph_store().query(**kwargs)
