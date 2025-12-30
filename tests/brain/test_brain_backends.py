import pytest

# pylint: disable=import-error
from third_party.autogpt.autogpt.core.brain.config import (
    BrainBackend,
    BrainSimulationConfig,
    WholeBrainConfig,
)
from modules.brain.backends import (
    brain_simulation_result_to_cycle,
    create_brain_backend,
    translate_agent_payload_for_brain_simulation,
)
from modules.brain.state import BrainCycleResult


def test_create_brain_backend_returns_whole_brain_instance():
    backend = create_brain_backend(
        BrainBackend.WHOLE_BRAIN,
        whole_brain_config=WholeBrainConfig(),
        brain_simulation_config=BrainSimulationConfig(),
    )
    assert hasattr(backend, "process_cycle")


def test_translate_agent_payload_for_brain_simulation_structures_inputs():
    payload = {
        "agent_id": "agent-1",
        "text": "Task context",
        "context": {"goal_focus": "ship feature", "progress": 0.6, "novelty": 0.4},
        "vision": [0.1, 0.2],
        "auditory": [0.3],
        "somatosensory": [0.5],
        "is_salient": True,
    }
    translated = translate_agent_payload_for_brain_simulation(payload)
    assert translated["perception"]["vision"] == [0.1, 0.2]
    assert translated["emotion"]["stimuli"]
    assert translated["curiosity"]["novelty"] == pytest.approx(0.4)


def test_brain_simulation_result_to_cycle_maps_core_fields():
    raw = {
        "time": 42.0,
        "network_state": {"spikes": [0.1, 0.2, 0.3]},
        "cognitive_state": {
            "perception": {"modalities": {"vision": {"intensity": 1.0}}},
            "emotion": {"primary": "happy", "intensity": 0.7},
            "curiosity": {"drive": 0.5, "stimulus": {"novelty": 0.4}},
            "personality": {"dynamic": {"openness": 0.8}},
            "decision": {"decision": "explore", "confidence": 0.9},
        },
    }
    cycle = brain_simulation_result_to_cycle(raw)
    assert isinstance(cycle, BrainCycleResult)
    assert cycle.intent.intention == "explore"
    assert cycle.emotion.primary.value == "happy"
    assert cycle.curiosity.drive == pytest.approx(0.5)
