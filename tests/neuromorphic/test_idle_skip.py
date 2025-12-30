import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.neuromorphic import SpikingNeuralNetwork


def test_idle_skip_reduces_processing_steps():
    events = [[0.0]] * 50 + [[1.1]]

    # Baseline network processes every timestep
    baseline = SpikingNeuralNetwork(n_neurons=1, threshold=1.0, reset=0.0, weights=[[0.0]])
    baseline.synapses.adapt = lambda *args, **kwargs: None
    baseline_outputs = baseline.run(events)
    baseline_steps = baseline.energy_usage
    baseline_spikes = [o for o in baseline_outputs if any(o[1])]

    # Idle-skipping network ignores the leading zeros
    idle = SpikingNeuralNetwork(
        n_neurons=1, threshold=1.0, reset=0.0, weights=[[0.0]], idle_skip=True
    )
    idle.synapses.adapt = lambda *args, **kwargs: None
    idle_outputs = idle.run(events)
    idle_steps = idle.energy_usage
    idle_spikes = [o for o in idle_outputs if any(o[1])]

    # Energy usage should be lower when skipping idle periods
    assert idle_steps < baseline_steps
    # Both networks should produce identical spike outputs
    assert baseline_spikes == idle_spikes
