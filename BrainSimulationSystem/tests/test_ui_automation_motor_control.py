"""Tests for UI automation integration with the motor control subsystem."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys


# Some environments omit optional dependencies; keep this test lightweight.
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


from BrainSimulationSystem.models.motor_control import MotorControlSystem  # noqa: E402


def test_motor_control_executes_ui_actions_from_intention_dict_in_dry_run():
    system = MotorControlSystem(
        {
            "backend": "heuristic",
            "action_dim": 3,
            "ui": {
                "enabled": True,
                "dry_run": True,
                "require_foreground": False,
                "min_interval_s": 0.0,
                "max_actions_per_minute": 0,
            },
        }
    )

    result = system.compute({"ui_actions": [{"action": "type_text", "text": "hello"}]})

    ui = result.get("ui")
    assert ui is not None
    assert ui.get("backend") == "noop"
    results = ui.get("results")
    assert isinstance(results, list) and results
    assert results[0].get("ok") is True


def test_motor_control_command_map_triggers_ui_actions_in_dry_run():
    system = MotorControlSystem(
        {
            "backend": "heuristic",
            "action_dim": 3,
            "ui": {
                "enabled": True,
                "dry_run": True,
                "require_foreground": False,
                "min_interval_s": 0.0,
                "max_actions_per_minute": 0,
                "command_map": {
                    "move_forward": [{"action": "press_key", "key": "w"}],
                },
            },
        }
    )

    result = system.compute("move_forward")
    ui = result.get("ui")
    assert ui is not None
    results = ui.get("results")
    assert isinstance(results, list) and results
    assert results[0].get("ok") is True

