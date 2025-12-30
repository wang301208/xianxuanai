"""Tests for the Dijkstra shortest path algorithm."""
from typing import Dict, List, Tuple
import math

from algorithms.graph.advanced.dijkstra import Dijkstra


def _build_graph() -> Dict[str, List[Tuple[str, float]]]:
    """Create a sample weighted undirected graph for testing."""
    return {
        "A": [("B", 1), ("C", 4)],
        "B": [("A", 1), ("C", 2), ("D", 5)],
        "C": [("A", 4), ("B", 2)],
        "D": [("B", 5)],
        "E": [],  # unreachable node
    }


def test_dijkstra_shortest_paths() -> None:
    graph = _build_graph()
    algo = Dijkstra()
    distances = algo.execute(graph, "A")

    assert distances["A"] == 0
    assert distances["B"] == 1
    assert distances["C"] == 3  # via A -> B -> C
    assert distances["D"] == 6  # via A -> B -> D


def test_dijkstra_unreachable_node() -> None:
    graph = _build_graph()
    algo = Dijkstra()
    distances = algo.execute(graph, "A")

    assert math.isinf(distances["E"])
