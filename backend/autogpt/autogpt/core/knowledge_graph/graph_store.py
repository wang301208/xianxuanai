"""Re-export GraphStore from the vendored AutoGPT implementation."""

from __future__ import annotations

try:  # pragma: no cover - import-time bridge
    from third_party.autogpt.autogpt.core.knowledge_graph.graph_store import *  # type: ignore
except Exception:  # pragma: no cover - minimal fallback
    from typing import Any, Dict, List, Optional

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

__all__ = ["GraphStore", "get_graph_store", "query_graph"]

