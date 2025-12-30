import os
import sys
from unittest.mock import Mock

import pytest

pytest.importorskip("numpy", reason="requires numpy for brain modules")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain import WholeBrainSimulation
from modules.brain.motor.actions import MotorExecutionResult, MotorPlan


def test_whole_brain_neuromorphic_cycle():
    brain = WholeBrainSimulation(neuromorphic=True)
    vision_stream = iter([[0.9, 0.1], [0.4, 0.6]])
    input_data = {
        "streams": {"vision": vision_stream},
        "sound": [0.5],
        "touch": [0.3, 0.7],
        "text": "good",
        "is_salient": True,
        "context": {"task": "explore"},
    }
    result = brain.process_cycle(input_data)
    assert result.metadata["executed_action"].startswith("executed")
    assert result.intent.plan
    assert result.metadata["policy"] == "heuristic"
    assert result.metadata["policy_metadata"]["confidence_calibrated"] is True
    assert result.metrics.get("cycle_index", 0.0) >= 1
    assert result.energy_used > 0
    assert "curiosity_drive" in result.metrics
    assert result.emotion.mood <= 1.0
    assert len(brain.perception_history) == 1
    assert "vision" in brain.last_perception.modalities
    assert brain.last_context.get("task") == "explore"
    assert brain.telemetry_log[-1]["modalities"]["vision"] in {"stream", "cached"}
    follow_up = {
        "streams": {"vision": vision_stream},
        "sound": [0.2],
        "touch": [0.6, 0.4],
        "context": {"task": "focus"},
    }
    second_result = brain.process_cycle(follow_up)
    assert len(brain.perception_history) == 2
    assert len(brain.decision_history) == 2
    assert brain.telemetry_log[-1]["cycle_index"] == brain.cycle_index
    assert second_result.intent.confidence <= 1.0
    assert brain.last_context.get("task") == "focus"
    assert brain.telemetry_log[-1]["cognitive_plan"]
    modulation = brain.get_strategy_modulation()
    assert modulation["curiosity_drive"] == brain.curiosity.drive


def test_energy_usage_varies_with_activity():
    brain = WholeBrainSimulation(neuromorphic=True)
    active = {"image": [1.0, 0.6], "sound": [0.9], "touch": [0.2, 0.8]}
    idle = {"image": [0.0, 0.0], "sound": [0.0], "touch": [0.0, 0.0]}
    energy_active = brain.process_cycle(active).energy_used
    idle_result = brain.process_cycle(idle)
    energy_idle = idle_result.energy_used
    assert energy_active > energy_idle
    assert idle_result.idle_skipped > 0


class _DummyMotor:
    def __init__(self) -> None:
        self.cerebellum = None
        self.precision_system = None

    def plan_movement(self, intention: str, parameters=None) -> MotorPlan:
        params = dict(parameters) if isinstance(parameters, dict) else {}
        return MotorPlan(intention=intention, stages=["dummy"], parameters=params, metadata={})

    def execute_action(self, plan: MotorPlan) -> MotorExecutionResult:
        return MotorExecutionResult(False, "failed", telemetry={}, error="simulated")

    def parse_feedback_metrics(self, feedback, base_reward=None):
        if isinstance(feedback, MotorExecutionResult):
            reward = base_reward if base_reward is not None else -0.5
            return {"success": float(feedback.success), "reward": float(reward)}
        return {}

    def train(self, feedback) -> None:  # pragma: no cover - not needed
        return None


def test_process_cycle_logs_exploration_outcomes():
    brain = WholeBrainSimulation(neuromorphic=False)
    dummy_motor = _DummyMotor()
    dummy_motor.cerebellum = brain.cerebellum
    dummy_motor.precision_system = brain.precision_motor
    brain.motor = dummy_motor

    def consume_once():
        if getattr(consume_once, "called", False):
            return {}
        consume_once.called = True
        return {
            "state_X": {
                "sample": {"goal_id": "explore:probe"},
                "metadata": {"priority": 3.0},
            }
        }

    brain.self_learning.consume_exploration_candidates = consume_once  # type: ignore[assignment]
    original_record = brain.self_learning.record_exploration_outcome
    mock_record = Mock(wraps=original_record)
    brain.self_learning.record_exploration_outcome = mock_record  # type: ignore[assignment]
    brain.self_learning.curiosity_driven_learning = lambda sample: {}  # type: ignore[assignment]

    brain.cognition.decide = lambda *args, **kwargs: {  # type: ignore[assignment]
        "intention": "observe",
        "confidence": 0.0,
        "plan": [],
        "weights": {},
        "policy_metadata": {},
    }

    brain.process_cycle({"image": [0.2, 0.1]})

    assert mock_record.called
    args, kwargs = mock_record.call_args
    assert args[0] == "state_X"
    assert args[1] is False
    metadata = brain.self_learning.memory["state_X"]["metadata"]
    assert metadata["failures"] >= 1
    assert metadata["priority"] > 1.0
