"""Tests for BrainSimulation persistence fallbacks."""

import json

from BrainSimulationSystem.brain_simulation import BrainSimulation


class _StubMemory:
    """Memory component without persistence attributes."""

    pass


class _StubDecision:
    """Decision component without persistence attributes."""

    pass


class _StubNetwork:
    """Network implementation lacking neuron/synapse collections."""

    pass


def test_save_and_load_state_with_stub_components(tmp_path) -> None:
    """Ensure save_state/load_state work when optional attributes are missing."""

    simulation = BrainSimulation()
    simulation.memory = _StubMemory()
    simulation.decision = _StubDecision()
    simulation.network = _StubNetwork()

    state_file = tmp_path / "state.json"

    # Should not raise even though the components lack the typical attributes
    simulation.save_state(str(state_file))

    # The saved file should exist and contain minimal cognitive state information
    with state_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["cognitive_state"]["memory"] == {
        "working_memory": {},
        "long_term_memory": {},
        "memory_strengths": {},
    }
    assert payload["cognitive_state"]["decision"] == {"action_values": {}, "decision_history": []}

    simulation.load_state(str(state_file))

    # Ensure fallback recreation produced functional adapters bound to the network
    assert hasattr(simulation.memory, "network")
    assert hasattr(simulation.decision, "network")
    assert simulation.memory.network is simulation.network
    assert simulation.decision.network is simulation.network


def test_load_state_recreates_network_and_adapters(tmp_path) -> None:
    """Loading a saved state should rebuild the runtime components."""

    config = {"network": {"learning_rules": {"stdp": {"enabled": True}}}}
    simulation = BrainSimulation(config)

    original_network = simulation.network
    original_perception = simulation.perception
    original_attention = simulation.attention
    original_memory = simulation.memory
    original_decision = simulation.decision

    state_file = tmp_path / "full_state.json"
    simulation.save_state(str(state_file))

    simulation.load_state(str(state_file))

    # A new network instance should have been created
    assert simulation.network is not original_network

    # Learning rules should now point at the new network
    assert simulation.learning_rules, "Learning rules should be recreated"
    assert all(rule.network is simulation.network for rule in simulation.learning_rules)

    # Cognition adapters should be re-bound to the recreated network
    assert simulation.perception is not original_perception
    assert simulation.attention is not original_attention
    assert simulation.memory is not original_memory
    assert simulation.decision is not original_decision
    for adapter in (simulation.perception, simulation.attention, simulation.memory, simulation.decision):
        assert getattr(adapter, "network", None) is simulation.network

    # The simulation should remain operable
    result = simulation.step({"sensory_data": [0.1], "decision_options": ["left", "right"]}, 1.0)
    assert result["time"] == simulation.current_time
