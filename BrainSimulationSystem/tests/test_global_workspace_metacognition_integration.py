"""Integration tests for the GlobalWorkspace metacognition controller."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from BrainSimulationSystem.brain_simulation import BrainSimulation  # noqa: E402


def test_metacognition_emits_workspace_and_commands_and_biases_next_step():
    simulation = BrainSimulation(
        {
            "metacognition": {"enabled": True},
        }
    )

    # Three options -> uniform softmax confidence ~ 1/3 < default threshold (0.35),
    # so metacognition should request more information and propose attention biases.
    result = simulation.step(
        {
            "decision_options": ["a", "b", "c"],
            "reward": 1.0,
        },
        dt=0.1,
    )

    meta = result["cognitive_state"].get("metacognition")
    assert isinstance(meta, dict)
    assert meta.get("enabled") is True

    commands = meta.get("commands")
    assert isinstance(commands, dict)
    assert commands.get("request_more_information") is True
    assert commands.get("learning_rate_scale", 1.0) > 1.0

    # Next step should apply pending attention overrides and decision exploration delta.
    _ = simulation.step(
        {
            "decision_options": ["a", "b"],
        },
        dt=0.1,
    )

    directives = simulation.last_attention_directives
    assert isinstance(directives, dict)
    weights = directives.get("modality_weights", {})
    assert isinstance(weights, dict)
    assert weights.get("language") == 1.0

    decision_params = getattr(simulation.decision, "params", {})
    assert isinstance(decision_params, dict)
    assert decision_params.get("exploration_rate", 0.0) > 0.1

