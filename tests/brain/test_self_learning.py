import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "modules" / "brain" / "self_learning.py"
spec = importlib.util.spec_from_file_location("modules.brain.self_learning", MODULE_PATH)
self_learning = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = self_learning
spec.loader.exec_module(self_learning)
SelfLearningBrain = self_learning.SelfLearningBrain


def test_curiosity_driven_learning_improves_prediction():
    brain = SelfLearningBrain()
    sample = {
        "state": "s1",
        "agent_id": "agent",
        "usage": {"cpu": 1.0, "memory": 2.0},
        "reward": 1.0,
    }

    before = brain.world_model.predict("agent")
    err_before = abs(before["cpu"] - 1.0) + abs(before["memory"] - 2.0)

    brain.curiosity_driven_learning(sample)

    after = brain.world_model.predict("agent")
    err_after = abs(after["cpu"] - 1.0) + abs(after["memory"] - 2.0)

    assert "s1" in brain.memory
    assert brain.memory["s1"]["sample"] == sample
    assert err_after < err_before


def test_exploration_candidates_surface_metadata():
    brain = SelfLearningBrain()
    brain.rejection_threshold = 1
    brain.error_threshold = 0.5
    brain.error_window = 2

    # First sample is accepted and initializes memory
    accepted_sample = {
        "state": "s_reject",
        "agent_id": "agent_reject",
        "usage": {"cpu": 1.0},
        "reward": 1.0,
    }
    brain.curiosity_driven_learning(accepted_sample)

    # Second sample is rejected which should trigger exploration flagging
    rejected_sample = {
        "state": "s_reject",
        "agent_id": "agent_reject",
        "usage": {"cpu": 1.0},
        "reward": -1.0,
    }
    brain.curiosity_driven_learning(rejected_sample)

    candidates = brain.consume_exploration_candidates()
    assert "s_reject" in candidates
    assert candidates["s_reject"]["metadata"]["rejections"] >= 1
    assert candidates["s_reject"]["metadata"]["flagged"] is False

    # Configure selector to always accept to accumulate error samples
    brain.selector.reward_threshold = -1.0

    high_error_first = {
        "state": "s_error",
        "agent_id": "agent_error",
        "usage": {"cpu": 10.0},
        "reward": 1.0,
    }
    brain.curiosity_driven_learning(high_error_first)

    high_error_second = {
        "state": "s_error",
        "agent_id": "agent_error",
        "usage": {"cpu": 0.0},
        "reward": 1.0,
    }
    brain.curiosity_driven_learning(high_error_second)

    candidates = brain.consume_exploration_candidates()
    assert "s_error" in candidates
    metadata = candidates["s_error"]["metadata"]
    assert metadata["smoothed_error"] > brain.error_threshold
    assert metadata["flagged"] is False


def test_record_exploration_outcome_updates_priority_and_flags():
    brain = SelfLearningBrain()
    sample = {
        "state": "state_outcome",
        "agent_id": "agent",
        "usage": {"cpu": 0.5},
        "reward": 0.1,
    }
    brain.curiosity_driven_learning(sample)
    metadata = brain.memory["state_outcome"]["metadata"]
    initial_priority = metadata["priority"]

    brain.record_exploration_outcome("state_outcome", success=False, metrics={"reward": -0.5})
    metadata = brain.memory["state_outcome"]["metadata"]
    assert metadata["attempts"] == 1
    assert metadata["failures"] == 1
    assert metadata["priority"] > initial_priority
    assert metadata["flagged"] is True
    assert "state_outcome" in brain.exploration_flags

    post_failure_priority = metadata["priority"]
    brain.consume_exploration_candidates()
    brain.record_exploration_outcome("state_outcome", success=True, metrics={"reward": 0.8})
    metadata = brain.memory["state_outcome"]["metadata"]
    assert metadata["successes"] == 1
    assert metadata["priority"] < post_failure_priority
    assert metadata["flagged"] is False


def test_consumed_candidates_prioritise_high_risk_states():
    brain = SelfLearningBrain()
    for idx, reward in enumerate([0.0, -1.0]):
        state = f"state_{idx}"
        brain.curiosity_driven_learning({
            "state": state,
            "agent_id": f"agent_{idx}",
            "usage": {"cpu": 0.2},
            "reward": reward,
        })
        brain.record_exploration_outcome(state, success=False, metrics={"reward": reward})

    candidates = brain.consume_exploration_candidates()
    assert list(candidates.keys())[0] == "state_1"
    high_priority = candidates["state_1"]["metadata"]["priority"]
    low_priority = candidates["state_0"]["metadata"]["priority"]
    assert high_priority >= low_priority


def test_register_knowledge_gap_generates_self_directed_goal():
    brain = SelfLearningBrain()
    state = brain.register_knowledge_gap(
        "Quantum Flux",
        context={"query": "quantum flux"},
        reason="missing-data",
        priority=2.5,
    )
    assert state is not None
    assert state.startswith("knowledge-gap:")

    candidates = brain.consume_exploration_candidates()
    assert state in candidates
    candidate = candidates[state]
    metadata = candidate["metadata"]
    assert metadata["origin"] == "knowledge-gap"
    assert metadata["concept"] == "Quantum Flux"
    assert metadata["priority"] >= 2.5
    assert metadata["flagged"] is False
    assert "missing-data" in metadata.get("reasons", [])
    sample = candidate["sample"]
    assert sample["state"] == state
    assert sample["goal_id"].startswith("self-study:")
    assert candidate["goal"].startswith("self-study:")
    assert state not in brain.knowledge_gap_flags
