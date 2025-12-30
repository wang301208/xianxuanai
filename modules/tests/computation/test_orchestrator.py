import pytest

from modules.brain.computation import ComputationOrchestrator


def test_execute_bubble_sort():
    orchestrator = ComputationOrchestrator()
    data = [5, 1, 4, 2, 8]
    result = orchestrator.execute_algorithm("bubble_sort", data, {})
    assert result == sorted(data)


def test_execute_binary_search():
    orchestrator = ComputationOrchestrator()
    data = [1, 3, 5, 7, 9]
    index = orchestrator.execute_algorithm("binary_search", data, {"target": 7})
    assert index == 3


def test_execute_invalid_algorithm():
    orchestrator = ComputationOrchestrator()
    with pytest.raises(ValueError):
        orchestrator.execute_algorithm("nonexistent", [], {})
