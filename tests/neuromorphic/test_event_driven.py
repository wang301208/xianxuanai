import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.neuromorphic import SpikingNeuralNetwork


def test_events_trigger_processing_only_when_present():
    weights = [[0.0, 1.0], [0.0, 0.0]]
    network = SpikingNeuralNetwork(n_neurons=2, threshold=1.0, reset=0.0, weights=weights)
    network.synapses.adapt = lambda *args, **kwargs: None

    # Single external event causes neuron 0 to spike at t=0 which in turn
    # schedules a synaptic event for neuron 1 at t=1.
    events = [(0, [1.1, 0.0])]
    spikes = network.run(events)

    assert spikes == [(0, [1, 0]), (1, [0, 1])]

