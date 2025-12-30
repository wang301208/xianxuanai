"""Integration tests for limbic/emotion modulation in BrainSimulation."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from BrainSimulationSystem.brain_simulation import BrainSimulation


def test_step_exposes_limbic_circuits_and_neuromodulators():
    simulation = BrainSimulation({"limbic": {"enabled": True}})

    result = simulation.step(
        {
            "decision_options": ["explore", "rest"],
            "reward": 1.0,
            "curiosity": {"threat": 0.8, "novelty": 0.4},
        },
        dt=0.1,
    )

    cognitive = result["cognitive_state"]
    decision = cognitive["decision"]
    assert "predicted_reward" in decision

    emotion = cognitive["emotion"]
    assert "limbic_circuits" in emotion
    assert "neuromodulators" in emotion

    network_state = result["network_state"]
    neuromodulators = network_state.get("neuromodulators")
    assert isinstance(neuromodulators, dict)
    assert "dopamine" in neuromodulators


def test_reward_prediction_error_is_recorded():
    simulation = BrainSimulation({"limbic": {"enabled": True}})

    result = simulation.step(
        {
            "decision_options": ["a", "b"],
            "reward": 0.7,
        },
        dt=0.1,
    )

    emotion = result["cognitive_state"]["emotion"]
    assert "reward_prediction_error" in emotion
