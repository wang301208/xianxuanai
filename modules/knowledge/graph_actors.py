from __future__ import annotations

"""Distributed helpers for graph ingestion and querying via Ray or Dask."""

import logging
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from backend.autogpt.autogpt.core.knowledge_graph.graph_store import get_graph_store
from backend.autogpt.autogpt.core.knowledge_graph.ontology import EntityType, RelationType

logger = logging.getLogger(__name__)

try:  # optional dependency
    import ray  # type: ignore

    _HAS_RAY = True
except Exception:  # pragma: no cover - optional dependency
    ray = None  # type: ignore
    _HAS_RAY = False


def _coerce_entity_type(value: Any) -> EntityType:
    if isinstance(value, EntityType):
        return value
    if isinstance(value, str):
        try:
            return EntityType(value.lower())
        except ValueError:
            logger.debug("Unknown entity type '%s'; defaulting to CONCEPT.", value)
    return EntityType.CONCEPT


def _coerce_relation_type(value: Any) -> RelationType:
    if isinstance(value, RelationType):
        return value
    if isinstance(value, str):
        try:
            return RelationType(value.lower())
        except ValueError:
            logger.debug("Unknown relation type '%s'; defaulting to RELATED_TO.", value)
    return RelationType.RELATED_TO


def _serialise_nodes(nodes: Sequence[Any]) -> list[Dict[str, Any]]:
    serialised: list[Dict[str, Any]] = []
    for node in nodes:
        node_id = getattr(node, "id", None)
        node_type = getattr(node, "type", None)
        props = getattr(node, "properties", {})
        if node_id is None:
            continue
        serialised.append(
            {
                "id": node_id,
                "type": node_type.value if isinstance(node_type, EntityType) else node_type,
                "properties": dict(props) if isinstance(props, Mapping) else dict(props or {}),
            }
        )
    return serialised


def _serialise_edges(edges: Sequence[Any]) -> list[Dict[str, Any]]:
    serialised: list[Dict[str, Any]] = []
    for edge in edges:
        source = getattr(edge, "source", None)
        target = getattr(edge, "target", None)
        relation = getattr(edge, "type", None)
        props = getattr(edge, "properties", {})
        if not source or not target:
            continue
        serialised.append(
            {
                "source": source,
                "target": target,
                "type": relation.value if isinstance(relation, RelationType) else relation,
                "properties": dict(props) if isinstance(props, Mapping) else dict(props or {}),
            }
        )
    return serialised


class _GraphIngestWorker:
    """Shared ingestion logic for Ray actors or direct calls."""

    def __init__(self) -> None:
        self._store = get_graph_store()

    def submit(self, batch: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        summary = {"nodes": 0, "edges": 0, "errors": 0}
        for item in batch or []:
            if not isinstance(item, Mapping):
                summary["errors"] += 1
                continue
            op = str(item.get("op") or item.get("operation") or "").lower()
            if not op:
                if "source" in item and "target" in item:
                    op = "edge"
                elif "entity_type" in item or "type" in item:
                    op = "node"
            try:
                if op == "node":
                    entity_type = _coerce_entity_type(item.get("entity_type") or item.get("type"))
                    node_id = str(item["id"])
                    properties = dict(item.get("properties", {}))
                    self._store.add_node(node_id, entity_type, **properties)
                    summary["nodes"] += 1
                elif op == "edge":
                    relation = _coerce_relation_type(item.get("relation_type") or item.get("type"))
                    source = str(item["source"])
                    target = str(item["target"])
                    properties = dict(item.get("properties", {}))
                    self._store.add_edge(source, target, relation, **properties)
                    summary["edges"] += 1
                else:
                    summary["errors"] += 1
            except Exception:
                logger.warning("Graph ingestion failed for item %s", item, exc_info=True)
                summary["errors"] += 1
        return summary


class _GraphQueryWorker:
    """Shared query logic for Ray actors or direct Dask functions."""

    def __init__(self) -> None:
        self._store = get_graph_store()

    def query(
        self,
        pattern: Optional[Mapping[str, Any]] = None,
        *,
        limit: Optional[int] = None,
    ) -> Dict[str, list[Dict[str, Any]]]:
        pattern = pattern or {}
        node_id = pattern.get("node_id") or pattern.get("id")
        entity_type_val = pattern.get("entity_type")
        relation_type_val = pattern.get("relation_type")
        entity_type = (
            _coerce_entity_type(entity_type_val) if entity_type_val is not None else None
        )
        relation_type = (
            _coerce_relation_type(relation_type_val) if relation_type_val is not None else None
        )
        result = self._store.query(
            node_id=node_id,
            entity_type=entity_type,
            relation_type=relation_type,
        )
        serialised = {
            "nodes": _serialise_nodes(result.get("nodes", [])),
            "edges": _serialise_edges(result.get("edges", [])),
        }
        if limit:
            serialised["nodes"] = serialised["nodes"][:limit]
            serialised["edges"] = serialised["edges"][:limit]
        return serialised


def ingest_batch(batch: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Ingest helper usable by synchronous callers or Dask tasks."""

    worker = _GraphIngestWorker()
    return worker.submit(batch)


def query_pattern(pattern: Optional[Mapping[str, Any]] = None, *, limit: Optional[int] = None) -> Dict[str, list[Dict[str, Any]]]:
    """Query helper usable by synchronous callers or Dask tasks."""

    worker = _GraphQueryWorker()
    return worker.query(pattern, limit=limit)


def create_ray_graph_ingest_actor(**ray_options: Any):
    """Instantiate a Ray actor for graph ingestion."""

    if not _HAS_RAY:
        raise RuntimeError("Ray is not available; install ray to use the graph ingest actor.")
    actor_cls = ray.remote(**ray_options)(_GraphIngestWorker)  # type: ignore[misc]
    handle = actor_cls.remote()  # type: ignore[no-any-return]

    class _IngestWrapper:
        def __init__(self, actor_handle):
            self._actor = actor_handle

        def submit(self, batch: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
            return ray.get(self._actor.submit.remote(batch))

        def submit_async(self, batch: Iterable[Mapping[str, Any]]):
            return self._actor.submit.remote(batch)

    return _IngestWrapper(handle)


def create_ray_graph_query_actor(**ray_options: Any):
    """Instantiate a Ray actor for graph queries."""

    if not _HAS_RAY:
        raise RuntimeError("Ray is not available; install ray to use the graph query actor.")
    actor_cls = ray.remote(**ray_options)(_GraphQueryWorker)  # type: ignore[misc]
    handle = actor_cls.remote()  # type: ignore[no-any-return]

    class _QueryWrapper:
        def __init__(self, actor_handle):
            self._actor = actor_handle

        def query(self, pattern: Optional[Mapping[str, Any]] = None, *, limit: Optional[int] = None):
            return ray.get(self._actor.query.remote(pattern or {}, limit=limit))

        def query_async(
            self,
            pattern: Optional[Mapping[str, Any]] = None,
            *,
            limit: Optional[int] = None,
        ):
            return self._actor.query.remote(pattern or {}, limit=limit)

    return _QueryWrapper(handle)


__all__ = [
    "ingest_batch",
    "query_pattern",
    "create_ray_graph_ingest_actor",
    "create_ray_graph_query_actor",
]
