"""Tests for the motor control subsystem and its integration with the brain simulation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

if "networkx" not in sys.modules:
    class _MockGraph:
        def __init__(self):
            self._nodes = {}
            self._edges = {}

        def add_node(self, node, **attr):
            self._nodes[node] = attr

        def nodes(self, data=False):
            if data:
                return list(self._nodes.items())
            return list(self._nodes.keys())

        def add_edge(self, u, v, **attr):
            self._edges.setdefault(u, {}).update({v: attr})
            self._edges.setdefault(v, {}).update({u: attr})

    sys.modules["networkx"] = SimpleNamespace(Graph=_MockGraph)

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from BrainSimulationSystem.brain_simulation import BrainSimulation  # noqa: E402
from BrainSimulationSystem.models.motor_control import MotorControlSystem  # noqa: E402


def test_motor_control_system_heuristic_backend_produces_command():
    system = MotorControlSystem({"backend": "heuristic", "action_dim": 3, "max_force": 2.0})
    result = system.compute("move_forward")

    assert "commands" in result
    commands = np.asarray(result["commands"], dtype=np.float32)
    assert commands.shape == (3,)
    assert np.linalg.norm(commands) > 0
    assert result["backend"] == "heuristic"


def test_brain_simulation_includes_motor_state_when_enabled():
    simulation = BrainSimulation({
        "motor": {
            "enabled": True,
            "controller": {
                "backend": "heuristic",
                "action_dim": 3,
                "max_force": 1.5,
            },
        }
    })

    inputs = {
        "sensory_data": [0.1, 0.2, 0.3],
        "decision_options": ["move_forward"],
        "motor_feedback": {"error_vector": [0.1, 0.0, 0.0]},
    }

    result = simulation.step(inputs, dt=0.1)
    motor_state = result["cognitive_state"].get("motor")

    assert motor_state is not None
    assert "commands" in motor_state
    assert len(motor_state["commands"]) == 3
