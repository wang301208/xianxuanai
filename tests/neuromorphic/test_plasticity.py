import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.neuromorphic import SpikingNeuralNetwork


def _build_network(weights=None):
    base_weights = weights if weights is not None else [[0.0, 0.0], [0.0, 0.0]]
    return SpikingNeuralNetwork(
        n_neurons=2,
        threshold=1.0,
        reset=0.0,
        decay=1.0,
        weights=base_weights,
    )


def test_stdp_potentiation():
    network = _build_network()
    inputs = [[1.1, 0.0], [0.0, 1.1]]  # pre neuron fires before post neuron
    network.run(inputs)
    assert network.synapses.weights[0][1] > 0.0


def test_stdp_depression():
    network = _build_network()
    inputs = [[0.0, 1.1], [1.1, 0.0]]  # post neuron fires before pre neuron
    network.run(inputs)
    assert network.synapses.weights[0][1] < 0.0


def test_oscillation_modulation_scales_learning_rate():
    inputs = [[1.1, 0.0], [0.0, 1.1]]
    baseline = _build_network()
    baseline.run(inputs)
    baseline_update = baseline.synapses.weights[0][1]

    enhanced = _build_network()
    enhanced.apply_modulation(
        {
            "amplitude_norm": 0.9,
            "synchrony_index": 0.9,
            "rhythmicity": 0.8,
            "plasticity_gate": 1.5,
        }
    )
    enhanced.run(inputs)
    enhanced_update = enhanced.synapses.weights[0][1]

    assert enhanced_update > baseline_update


def test_low_oscillation_activity_suppresses_learning():
    inputs = [[1.1, 0.0], [0.0, 1.1]]
    baseline = _build_network()
    baseline.run(inputs)
    baseline_update = baseline.synapses.weights[0][1]

    suppressed = _build_network()
    suppressed.apply_modulation(
        {
            "amplitude_norm": 0.05,
            "synchrony_index": 0.05,
            "rhythmicity": 0.0,
            "plasticity_gate": 0.1,
        }
    )
    suppressed.run(inputs)
    suppressed_update = suppressed.synapses.weights[0][1]

    assert suppressed_update < baseline_update
