"""Multi-hop association utilities.

The algorithm implemented here performs a breadth-first search over a simple
knowledge graph in order to connect concepts from potentially different
domains.  It serves as a reference for how a reasoning step could actively
combine disparate pieces of knowledge during inference.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List

Graph = Dict[str, Iterable[str]]


class MultiHopAssociator:
    """Perform multi-hop association over a knowledge graph."""

    def __init__(self, graph: Graph):
        self.graph = graph

    def find_path(self, start: str, goal: str) -> List[str]:
        """Return a list of nodes connecting ``start`` to ``goal``.

        The function implements a straightforward breadth-first search and
        returns the first path found.  If no path is present, an empty list is
        returned.
        """

        queue = deque([(start, [start])])
        visited = {start}
        while queue:
            node, path = queue.popleft()
            if node == goal:
                return path
            for neighbor in self.graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return []
