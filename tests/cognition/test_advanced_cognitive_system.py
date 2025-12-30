import os
import sys

import pytest

# Ensure repository root on import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.cognition.advanced import AdvancedCognitiveSystem


def test_working_memory_capacity():
    system = AdvancedCognitiveSystem(memory_capacity=3)
    for item in ["a", "b", "c", "d"]:
        system.store_in_memory(item)
    memory = system.retrieve_memory()
    assert len(memory) == 3
    assert memory == ["b", "c", "d"]


def test_executive_control_operations():
    system = AdvancedCognitiveSystem()
    assert system.conflict_monitoring(["go", "go"]) is True
    assert system.cognitive_flexibility(["t1", "t1", "t2"]) == "switch"
    assert system.response_inhibition(True) is False
    system.update_working_memory("item1")
    assert "item1" in system.retrieve_memory()


def test_decision_making_models():
    system = AdvancedCognitiveSystem()
    decision_dual = system.make_decision({"A": 0.9, "B": 0.1}, model="dual")
    assert decision_dual == "A"
    decision_rl = system.make_decision({"A": 1.0, "B": 0.5}, model="rl")
    assert decision_rl == "A"


def test_metacognitive_confidence():
    system = AdvancedCognitiveSystem()
    confidence = system.assess_confidence("A", 0.7)
    assert confidence == pytest.approx(0.7, rel=1e-6)
