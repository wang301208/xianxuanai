import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modules.brain.collaboration import MultiAgentCoordinator, NeuralMessageBus


def test_coordinator_aggregates_results() -> None:
    coordinator = MultiAgentCoordinator(NeuralMessageBus())

    coordinator.register_agent("a", lambda payload: payload["v"] * 2)
    coordinator.register_agent("b", lambda payload: payload["v"] + 1)

    coordinator.declare_task("t1", "a", {"v": 2})
    coordinator.declare_task("t2", "b", {"v": 3})

    coordinator.assign_task("t1")
    coordinator.assign_task("t2")

    results = coordinator.synchronize()
    assert results == {"t1": 4, "t2": 4}

    state = coordinator.world_model.get_state()
    assert "t1" in state["tasks"] and "t2" in state["tasks"]


def test_conflict_resolution_keeps_first_result() -> None:
    coordinator = MultiAgentCoordinator(NeuralMessageBus())

    coordinator.register_agent("a", lambda payload: "first")
    coordinator.declare_task("t1", "a", {})
    coordinator.assign_task("t1")

    # Publish a conflicting result from another agent
    coordinator.bus.publish(
        {"target": "coordinator", "task_id": "t1", "result": "second", "agent_id": "b"}
    )

    results = coordinator.synchronize()
    assert results["t1"] == "first"

    actions = coordinator.world_model.get_state()["actions"]
    assert actions == [{"agent_id": "a", "action": "completed t1"}]
