from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 提供最小的 NeuralNetwork 桩对象，避免依赖真实实现
import BrainSimulationSystem.core  # type: ignore  # noqa: E402

if "BrainSimulationSystem.core.network" not in sys.modules:
    class _StubNetwork:  # pragma: no cover - 简化桩对象
        pass

    stub_module = type(sys)("BrainSimulationSystem.core.network")
    stub_module.NeuralNetwork = _StubNetwork  # type: ignore[attr-defined]
    sys.modules["BrainSimulationSystem.core.network"] = stub_module

from BrainSimulationSystem.models.cognitive_controller import CognitiveController  # noqa: E402


def _build_controller() -> CognitiveController:
    params = {
        "attention": {"bottom_up_weight": 0.6, "top_down_weight": 0.6},
        "working_memory": {"capacity": 5},
    }
    return CognitiveController(params=params)


def test_attention_allocation_includes_bottom_up_and_top_down_bias():
    controller = _build_controller()

    result = controller.process(
        {
            "sensory_input": {"vision": 0.9, "auditory": 0.2, "language": 0.1},
            "task_goal": "language comprehension",
            "control_signals": {"language": {"attention": 0.9}},
        }
    )

    allocation = result["attention_allocation"]
    weights = allocation["weights"]

    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert weights["language"] > weights.get("auditory", 0.0)
    assert "language" in allocation["focus"]
    assert controller.workspace["_attention"]["language"] == weights["language"]


def test_attention_workspace_reacts_to_salience_shift():
    controller = _build_controller()

    controller.process({"sensory_input": {"vision": 0.2, "auditory": 0.4}})
    result = controller.process({"sensory_input": {"vision": 0.95, "auditory": 0.1}})

    weights = result["attention_allocation"]["weights"]
    assert weights.get("vision", 0.0) > weights.get("auditory", 0.0)
    assert controller.workspace["_attention_focus"]
    assert controller.workspace["_attention_focus"][0] == "vision"

