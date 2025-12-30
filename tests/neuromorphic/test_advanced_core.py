import pytest

from modules.brain.neuromorphic.advanced_core import (
    AdvancedNeuromorphicCore,
    NetworkTopologyGenerator,
    NeuromorphicHardwareBackend,
)


def test_neuron_instantiation():
    core = AdvancedNeuromorphicCore()
    neurons = [
        core.create_neuron("adaptive_exponential"),
        core.create_neuron("fast_spiking"),
        core.create_neuron("dopamine"),
        core.create_neuron("chattering"),
    ]
    # Ensure each neuron can step without error
    for neuron in neurons:
        neuron.step(1.0)


def test_synapse_instantiation():
    core = AdvancedNeuromorphicCore()
    synapses = [
        core.create_synapse("ampa"),
        core.create_synapse("gaba"),
        core.create_synapse("dopamine"),
        core.create_synapse("cholinergic"),
    ]
    outputs = [s.transmit(1) for s in synapses]
    assert len(outputs) == 4


def test_topology_generator():
    gen = NetworkTopologyGenerator(seed=42)
    for topo in ["small_world", "scale_free", "modular"]:
        matrix = gen.generate(10, topo)
        assert len(matrix) == 10
        assert len(matrix[0]) == 10


def test_backend_selection():
    core = AdvancedNeuromorphicCore()
    backend = core.select_backend("loihi", optimization_level=2)
    assert isinstance(backend, NeuromorphicHardwareBackend)
    assert backend.target_chip == "loihi"
    with pytest.raises(ValueError):
        core.select_backend("unknown")
