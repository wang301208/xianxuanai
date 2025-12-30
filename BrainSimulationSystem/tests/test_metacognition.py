from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from BrainSimulationSystem.models.cognitive_controller import CognitiveController  # noqa: E402


class DummyNetwork:
    pass


def build_controller(extra_params: dict | None = None) -> CognitiveController:
    params = {
        "attention": {},
        "working_memory": {},
        "self_model": {"module": {"introspection_interval": 0.0}},
    }
    if extra_params:
        params.update(extra_params)
    return CognitiveController(DummyNetwork(), params)


def test_self_monitor_generates_summary_and_alerts():
    controller = build_controller()

    result = controller.process(
        {
            "sensory_input": {"vision": 0.5, "auditory": 0.2},
            "task_goal": "language analysis",
        }
    )

    evaluation = result.get("self_evaluation", {})
    assert evaluation.get("summary"), "Self evaluation should include textual summary"
    assert "high_uncertainty" in evaluation.get("alerts", []), "High uncertainty should trigger alert"
    assert "attention" in controller.control_signals, "Reflection should adjust attention control signals"
    assert controller.control_signals["attention"]["attention"] >= 0.75


def test_reflection_history_tracks_reports():
    controller = build_controller()

    controller.process({"sensory_input": {"vision": 0.1}})
    reflections = controller.meta_context.get("reflections", [])

    assert reflections, "Reflections history should record self-monitor outputs"
    assert reflections[-1]["alerts"], "Reflections should capture alerts from introspection"
