"""Tests for the Kruskal Minimum Spanning Tree algorithm."""
from typing import Dict, List, Tuple

import pytest

from algorithms.graph.advanced.mst import KruskalMST


def _build_connected_graph() -> Dict[str, List[Tuple[str, float]]]:
    return {
        "A": [("B", 1), ("C", 5)],
        "B": [("A", 1), ("C", 4), ("D", 2)],
        "C": [("A", 5), ("B", 4), ("D", 1)],
        "D": [("B", 2), ("C", 1)],
    }


def _build_negative_graph() -> Dict[str, List[Tuple[str, float]]]:
    return {
        "A": [("B", -1), ("C", 4)],
        "B": [("A", -1), ("C", 2), ("D", 3)],
        "C": [("A", 4), ("B", 2), ("D", -2)],
        "D": [("B", 3), ("C", -2)],
    }


def _build_disconnected_graph() -> Dict[str, List[Tuple[str, float]]]:
    return {
        "A": [("B", 1)],
        "B": [("A", 1)],
        "C": [],
    }


def test_kruskal_mst_connected_graph() -> None:
    graph = _build_connected_graph()
    algo = KruskalMST()
    edges, total = algo.execute(graph)

    expected_edges = {
        frozenset({"A", "B"}),
        frozenset({"C", "D"}),
        frozenset({"B", "D"}),
    }
    assert {frozenset({u, v}) for u, v, _ in edges} == expected_edges
    assert total == 4


def test_kruskal_mst_negative_weights() -> None:
    graph = _build_negative_graph()
    algo = KruskalMST()
    edges, total = algo.execute(graph)

    expected_edges = {
        frozenset({"C", "D"}),
        frozenset({"A", "B"}),
        frozenset({"B", "C"}),
    }
    assert {frozenset({u, v}) for u, v, _ in edges} == expected_edges
    assert total == -1


def test_kruskal_mst_disconnected_graph() -> None:
    graph = _build_disconnected_graph()
    algo = KruskalMST()
    with pytest.raises(ValueError):
        algo.execute(graph)
