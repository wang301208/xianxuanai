"""Simple commonsense reasoning utilities for the knowledge graph."""
from __future__ import annotations

from typing import Dict, List, Tuple

from .graph_store import GraphStore, get_graph_store
from .ontology import RelationType


def infer_potential_relations(
    graph: GraphStore | None = None,
) -> List[Tuple[str, str, str]]:
    """Infer potential relations via two-hop paths.

    For each path A -> B -> C where no direct edge A -> C exists, suggest a
    potential relation between A and C. The returned tuples contain the source,
    target and the relation sequence discovered.
    """

    graph = graph or get_graph_store()
    data = graph.query()
    edges = data["edges"]
    adjacency: Dict[str, List[Tuple[str, RelationType]]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source, []).append((edge.target, edge.type))

    suggestions: List[Tuple[str, str, str]] = []
    for a, neighbors in adjacency.items():
        for mid, rel1 in neighbors:
            for c, rel2 in adjacency.get(mid, []):
                if any(e.source == a and e.target == c for e in edges):
                    continue
                suggestions.append((a, c, f"{rel1.value}->{rel2.value}"))
    return suggestions
