import os
import sys

import pytest

# Ensure repository root is on PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.self_model import SelfModel, SelfPlanner
from backend.memory.long_term import LongTermMemory
class DummyBus:
    def __init__(self) -> None:
        self._subs = []

    def subscribe(self, _topic, handler):
        self._subs.append(handler)

    def publish(self, _topic, event):
        for h in self._subs:
            h(event)


def test_assess_state_and_event():
    data = {"cpu": 1.0, "memory": 2.0}
    env_pred = {"avg_cpu": 2.0, "avg_memory": 3.0}
    memory = LongTermMemory(":memory:")
    memory.add("decision_outcome", "decision: previous success")
    self_model = SelfModel(memory)

    metrics, summary = self_model.assess_state(data, env_pred, "idle")

    assert metrics["cpu"] == pytest.approx(0.8)
    assert metrics["memory"] == pytest.approx(1.7)
    assert summary in self_model.history
    assert "previous success" in summary

    bus = DummyBus()
    received = []
    bus.subscribe("agent.self_awareness", lambda e: received.append(e))
    bus.publish("agent.self_awareness", {"agent": "test", "summary": summary})

    assert received and received[0]["summary"] == summary

    narratives = list(self_model._memory.get("self_narrative"))
    assert narratives and any("mood=" in n for n in narratives)


def test_update_state_from_events():
    model = SelfModel()
    model.update_state([
        "goal: build features",
        "subgoal: design module",
        "capability: planning (0.9)",
        "success",
    ])

    assert model._self_state["goals"][0]["goal"] == "build features"
    assert "design module" in model._self_state["goals"][0]["subgoals"]
    assert model._self_state["capabilities"]["planning"] == pytest.approx(0.9)
    assert model._self_state["mood"] == "satisfied"


def test_introspect_exposes_last_summary():
    data = {"cpu": 1.0, "memory": 2.0}
    env_pred = {"avg_cpu": 0.5, "avg_memory": 1.0}
    model = SelfModel()
    result = model.introspect(data, env_pred, "idle")
    assert "summary" in result
    assert model.last_summary == result["summary"]


def test_self_planner_revises_goals():
    memory = LongTermMemory(":memory:")
    model = SelfModel(memory)
    planner = SelfPlanner(model)
    planner.set_goal("build features", ["design"])
    new_goals = planner.revise_goals()

    assert new_goals == ["goals=build features:design; capabilities=none"]
    stored = list(memory.get("planning"))
    assert stored and stored[0].endswith("[revised]")
