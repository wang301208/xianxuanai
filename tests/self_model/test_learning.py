import os
import sys

import pytest

# Ensure repository root is on PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.self_model import SelfModel


def test_record_outcome_updates_capability_and_dataset():
    model = SelfModel()
    model.record_outcome("search", 0.0)
    first = model._self_state["capabilities"]["search"]
    model.record_outcome("search", 0.5)
    second = model._self_state["capabilities"]["search"]
    model.record_outcome("search", 1.0)
    third = model._self_state["capabilities"]["search"]

    assert first < second < third
    assert len(model.dataset) == 3
    assert model.dataset[-1] == ("search", 1.0)


def test_generate_subgoals_from_capabilities():
    model = SelfModel()
    model.add_goal("improve skills")
    for _ in range(5):
        model.record_outcome("planning", 1.0)
    subgoals = model.generate_subgoals()

    assert any("use planning" == sg for sg in subgoals)
    assert "use planning" in model._self_state["goals"][-1]["subgoals"]
