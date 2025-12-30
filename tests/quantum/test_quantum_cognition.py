import os
import sys

import numpy as np
import pytest

# Ensure the repository root is on the import path when the test module is
# executed in isolation.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.quantum.quantum_cognition import (
    EntanglementNetwork,
    QuantumCognition,
    SuperpositionState,
)
from modules.brain.quantum.quantum_memory import QuantumMemory


def test_superposition_probabilities():
    state = SuperpositionState({"0": 1 / np.sqrt(2), "1": 1 / np.sqrt(2)})
    qc = QuantumCognition()
    probs = qc.evaluate_probabilities(state)
    assert probs["0"] == pytest.approx(0.5, rel=1e-6)
    assert probs["1"] == pytest.approx(0.5, rel=1e-6)


def test_entangled_density_matrix_probabilities():
    network = EntanglementNetwork()
    bell_state = network.create_bell_pair()
    density = bell_state.density_matrix()
    qc = QuantumCognition()
    probs = qc.evaluate_probabilities(density)
    assert probs["00"] == pytest.approx(0.5, rel=1e-6)
    assert probs["11"] == pytest.approx(0.5, rel=1e-6)
    assert probs["01"] == pytest.approx(0.0, abs=1e-6)
    assert probs["10"] == pytest.approx(0.0, abs=1e-6)


def test_make_decision_interference():
    qc = QuantumCognition()
    choice, probs = qc.make_decision(
        {
            "A": [0.5, 0.5],
            "B": [0.5, -0.5],
        }
    )
    assert choice == "A"
    assert probs["A"] == pytest.approx(1.0, abs=1e-6)
    assert probs["B"] == pytest.approx(0.0, abs=1e-6)


def test_entangle_concepts_decoherence():
    network = EntanglementNetwork()
    density = network.entangle_concepts("cat", "hat", decoherence=0.5)
    qc = QuantumCognition()
    probs = qc.evaluate_probabilities(density)
    assert probs["00"] == pytest.approx(0.5, rel=1e-6)
    assert density[0, 3] == pytest.approx(0.25, rel=1e-6)


def test_quantum_memory_superposition():
    memory = QuantumMemory()
    memory.store("A", SuperpositionState({"0": 1}))
    memory.store("B", SuperpositionState({"1": 1}))
    combined = memory.superposition({"A": 1 / np.sqrt(2), "B": 1 / np.sqrt(2)})
    qc = QuantumCognition()
    probs = qc.evaluate_probabilities(combined)
    assert probs["0"] == pytest.approx(0.5, rel=1e-6)
    assert probs["1"] == pytest.approx(0.5, rel=1e-6)
