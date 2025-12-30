"""Tests for the world model module."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load(name: str, path: str):
    abs_path = Path(__file__).resolve().parents[2] / path
    spec = importlib.util.spec_from_file_location(name, str(abs_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Create a package-like structure for ``backend`` so modules can be imported
backend_pkg = types.ModuleType("backend")
backend_pkg.__path__ = []
sys.modules["backend"] = backend_pkg

wm_module = _load("backend.world_model", "backend/world_model/__init__.py")
backend_pkg.world_model = wm_module
WorldModel = wm_module.WorldModel


def test_state_updates_and_predictions():
    wm = WorldModel()

    # Add a task and update resources twice to exercise the learning component
    wm.add_task("task1", {"description": "test"})
    wm.update_resources("agent1", {"cpu": 1.0, "memory": 1.0})
    wm.update_resources("agent1", {"cpu": 3.0, "memory": 3.0})
    wm.record_action("agent1", "run")
    wm.record_action(
        "agent1",
        "retry",
        status="failed",
        result="timeout",
        error="timeout",
        metrics={"duration": 1.5},
        retries=1,
        metadata={"task_id": "task1"},
    )

    state = wm.get_state()
    assert "task1" in state["tasks"]
    assert state["resources"]["agent1"]["cpu"] == 3.0
    assert state["actions"][0]["action"] == "run"
    extended = state["actions"][1]
    assert extended["status"] == "failed"
    assert extended["metrics"]["duration"] == 1.5
    assert extended["metadata"]["task_id"] == "task1"

    # The prediction should be the moving average of the two updates
    pred_agent = wm.predict("agent1")
    assert pred_agent["cpu"] == 2.0
    assert pred_agent["memory"] == 2.0

    pred_all = wm.predict()
    assert pred_all["avg_cpu"] == 2.0
    assert pred_all["avg_memory"] == 2.0


def test_world_model_simulation_and_intrinsic_support():
    wm = WorldModel()

    wm.update_competence("planning", 0.3, source="test")
    wm.record_opportunity("Explore robotics frontier", weight=0.7, metadata={"origin": "test"})

    trajectory = wm.simulate(
        [
            {
                "domain": "planning",
                "agent_id": "self",
                "estimated_load": 4.0,
                "learning_rate": 0.15,
                "discover": "robotics frontier",
            }
        ],
        horizon=1,
    )

    assert trajectory, "Simulation should return at least one projected state"
    projected_state = trajectory[-1]
    competence_block = projected_state.get("competence", {}).get("planning", {})
    assert competence_block.get("score", 0.0) > 0.3

    gaps = wm.knowledge_gaps(threshold=0.5)
    assert "planning" in gaps

    targets = wm.suggest_learning_targets(limit=1, threshold=0.9)
    assert targets == ["planning"]

    opportunities = wm.suggest_opportunities(limit=1, threshold=0.5)
    assert opportunities and opportunities[0]["topic"] == "Explore robotics frontier"

