from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from BrainSimulationSystem.brain_simulation import BrainSimulation


class DummySynapse:
    def __init__(self, pre: int, post: int, weight: float) -> None:
        self.pre_neuron_id = pre
        self.post_neuron_id = post
        self.current_weight = weight
        self.weight = weight


class DummySynapseManager:
    def __init__(self, synapses) -> None:
        self.synapses = {idx: syn for idx, syn in enumerate(synapses)}


def build_simulation(pre: int, post: int, weight: float, *, potentiation: float, decay: float):
    sim = BrainSimulation.__new__(BrainSimulation)
    synapse = DummySynapse(pre, post, weight)
    network = SimpleNamespace(
        synapse_manager=DummySynapseManager([synapse]),
        _column_neuron_to_global={},
        _runtime_synapses=[],
    )
    sim.network = network
    sim.config = {"hebbian": {}}
    sim.hebbian_enabled = True
    sim.hebbian_params = {
        "potentiation": potentiation,
        "decay": decay,
        "max_weight": 5.0,
        "min_weight": -5.0,
    }
    sim.hebbian_metrics = {}
    sim.logger = logging.getLogger("HebbianTest")
    return sim, synapse


def test_hebbian_strengthens_coactive_synapse():
    sim, synapse = build_simulation(
        pre=1,
        post=2,
        weight=0.5,
        potentiation=0.2,
        decay=0.0,
    )

    spikes = {"spikes": [{"neuron_global": 1}, {"neuron_global": 2}]}
    stats = sim._apply_hebbian_plasticity(spikes, dt=1.0)

    assert pytest.approx(synapse.current_weight) == 0.7
    assert stats["coactive_pairs"] == 1
    assert stats["updated_synapses"] == 1


def test_hebbian_decay_reduces_inactive_synapse():
    sim, synapse = build_simulation(
        pre=1,
        post=2,
        weight=1.0,
        potentiation=0.0,
        decay=0.1,
    )

    stats = sim._apply_hebbian_plasticity({"spikes": []}, dt=1.0)

    assert pytest.approx(synapse.current_weight) == 0.9
    assert stats["updated_synapses"] == 1
    assert "coactive_pairs" not in stats or stats["coactive_pairs"] == 0

