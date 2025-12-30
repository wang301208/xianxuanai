import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.neuromorphic import SpikingNeuralNetwork, latency_encode


def test_latency_encode_basic():
    events = latency_encode([0.0, 0.5, 1.0])
    expected = [
        (0.0, [0, 0, 1]),
        (0.5, [0, 1, 0]),
        (1.0, [1, 0, 0]),
    ]
    assert events == expected


def test_run_with_latency_encoding():
    network = SpikingNeuralNetwork(
        n_neurons=1, decay=1.0, threshold=0.5, reset=0.0, weights=[[0.0]]
    )
    network.synapses.adapt = lambda *args, **kwargs: None
    outputs = network.run([[1.0], [0.0]], encoder=latency_encode)
    assert outputs == [(0, [1]), (2.0, [1])]
