import time

from agbenchmark.utils.dependencies.graphs import is_circular


def test_is_circular():
    cyclic_graph = {
        "nodes": [
            {"id": "A", "data": {"category": []}},
            {"id": "B", "data": {"category": []}},
            {"id": "C", "data": {"category": []}},
            {"id": "D", "data": {"category": []}},  # New node
        ],
        "edges": [
            {"from": "A", "to": "B"},
            {"from": "B", "to": "C"},
            {"from": "C", "to": "D"},
            {"from": "D", "to": "A"},  # This edge creates a cycle
        ],
    }

    result = is_circular(cyclic_graph)
    assert result is not None, "Expected a cycle, but none was detected"
    assert all(
        (
            (result[i], result[i + 1])
            in [(x["from"], x["to"]) for x in cyclic_graph["edges"]]
        )
        for i in range(len(result) - 1)
    ), "The detected cycle path is not part of the graph's edges"


def test_is_not_circular():
    acyclic_graph = {
        "nodes": [
            {"id": "A", "data": {"category": []}},
            {"id": "B", "data": {"category": []}},
            {"id": "C", "data": {"category": []}},
            {"id": "D", "data": {"category": []}},  # New node
        ],
        "edges": [
            {"from": "A", "to": "B"},
            {"from": "B", "to": "C"},
            {"from": "C", "to": "D"},
            # No back edge from D to any node, so it remains acyclic
        ],
    }

    assert is_circular(acyclic_graph) is None, "Detected a cycle in an acyclic graph"


def test_is_circular_performance():
    """Ensure cycle detection scales sub-quadratically for larger graphs."""
    n = 2000
    nodes = [{"id": str(i), "data": {"category": []}} for i in range(n)]
    edges = [{"from": str(i), "to": str(i + 1)} for i in range(n - 1)]
    edges.append({"from": str(n - 1), "to": "0"})
    graph = {"nodes": nodes, "edges": edges}

    start = time.perf_counter()
    assert is_circular(graph) is not None
    elapsed = time.perf_counter() - start
    # The previous implementation took ~0.09s for n=2000
    assert elapsed < 0.05
