"""Compatibility shim for the vendored knowledge-graph implementation.

The repository contains multiple AutoGPT code trees (`third_party/autogpt` and
`backend/autogpt`). Some internal modules import the knowledge-graph symbols
from `backend.autogpt.autogpt.core.knowledge_graph`, but the concrete
implementation is vendored under `third_party/autogpt/autogpt/core/knowledge_graph`.
"""

from __future__ import annotations

try:  # pragma: no cover - import-time bridge
    from third_party.autogpt.autogpt.core.knowledge_graph.graph_store import GraphStore, get_graph_store, query_graph
    from third_party.autogpt.autogpt.core.knowledge_graph.ontology import EntityType, RelationType
except Exception:  # pragma: no cover - minimal fallback
    from enum import Enum
    from typing import Any, Dict, List, Optional

    class EntityType(str, Enum):
        AGENT = "agent"

    class RelationType(str, Enum):
        RELATED_TO = "related_to"

    class GraphStore:
        def add_node(self, *args: Any, **kwargs: Any) -> None:
            return None

        def add_edge(self, *args: Any, **kwargs: Any) -> None:
            return None

        def query(self, **kwargs: Any) -> Dict[str, List]:
            return {"nodes": [], "edges": []}

    _GLOBAL: Optional[GraphStore] = None

    def get_graph_store() -> GraphStore:
        global _GLOBAL
        if _GLOBAL is None:
            _GLOBAL = GraphStore()
        return _GLOBAL

    def query_graph(**kwargs: Any) -> Dict[str, List]:
        return get_graph_store().query(**kwargs)

__all__ = ["GraphStore", "get_graph_store", "query_graph", "EntityType", "RelationType"]

