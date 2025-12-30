from __future__ import annotations

from dataclasses import dataclass

from BrainSimulationSystem.core.learning_rules.neuromodulated_stdp import NeuromodulatedSTDPLearning
from BrainSimulationSystem.core.synapse_manager import SynapseManager


@dataclass
class DummySynapse:
    post_neuron_id: int
    last_neuromodulators: dict | None = None

    def update(self, dt, current_time, post_voltage, astrocyte_activity=0.0, neuromodulators=None):
        self.last_neuromodulators = neuromodulators
        return 0.0


def test_synapse_manager_threads_neuromodulators():
    manager = SynapseManager(config={})
    synapse = DummySynapse(post_neuron_id=1)
    manager.synapses = {0: synapse}

    manager.update_all_synapses(
        dt=1.0,
        current_time=0.0,
        neuron_voltages={1: -65.0},
        astrocyte_activities={},
        neuromodulators={"dopamine": 0.9},
    )

    assert synapse.last_neuromodulators == {"dopamine": 0.9}


class _Synapse:
    def __init__(self, pre: int, post: int, weight: float):
        self.pre_neuron_id = pre
        self.post_neuron_id = post
        self.weight = weight

    def set_weight(self, value: float) -> None:
        self.weight = float(value)


class _Network:
    def __init__(self):
        self.neurons = {0: object(), 1: object()}
        self.synapses = {(0, 1): _Synapse(0, 1, 0.5)}


def test_neuromodulated_stdp_increases_weight_with_dopamine_gate():
    network = _Network()
    rule = NeuromodulatedSTDPLearning(
        network,
        params={
            "learning_rate": 1.0,
            "a_plus": 1.0,
            "a_minus": 0.0,
            "tau_plus": 20.0,
            "tau_minus": 20.0,
            "tau_eligibility": 200.0,
            "dopamine_baseline": 0.0,
            "weight_min": 0.0,
            "weight_max": 1.0,
        },
    )

    # Pre spike builds eligibility precursor (pre-trace).
    rule.update({"spikes": [0], "neuromodulators": {"dopamine": 0.0}}, dt=1.0)

    # Post spike paired with dopamine gates a positive weight update.
    rule.update({"spikes": [1], "neuromodulators": {"dopamine": 1.0}}, dt=1.0)

    syn = network.synapses[(0, 1)]
    assert syn.weight > 0.5

