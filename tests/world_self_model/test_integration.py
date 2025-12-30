import os
import sys

import pytest

# Ensure the repository root is on the PYTHONPATH so backend modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.world_model import WorldModel
from backend.self_model import SelfModel


def test_world_and_self_model_integration():
    resources = {
        "agent1": {"cpu": 1.0, "memory": 2.0},
        "agent2": {"cpu": 3.0, "memory": 4.0},
    }
    world = WorldModel()
    env_pred = world.predict(resources)
    self_model = SelfModel()
    corrected = self_model.estimate(resources["agent1"], env_pred)

    assert env_pred == {"avg_cpu": 2.0, "avg_memory": 3.0}
    assert corrected["cpu"] == pytest.approx(0.8)
    assert corrected["memory"] == pytest.approx(1.7)
