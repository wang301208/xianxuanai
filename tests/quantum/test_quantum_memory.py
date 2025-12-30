import os
import sys

import numpy as np
import pytest

# Ensure repo root in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.quantum.quantum_cognition import QuantumCognition, SuperpositionState
from modules.brain.quantum.quantum_memory import QuantumMemory


def test_store_and_retrieve():
    memory = QuantumMemory()
    state = SuperpositionState({"0": 1})
    memory.store("state", state)
    assert memory.retrieve("state") == state


def test_superposition_retrieval():
    memory = QuantumMemory()
    memory.store("A", SuperpositionState({"0": 1}))
    memory.store("B", SuperpositionState({"1": 1}))
    combined = memory.superposition({"A": 1 / np.sqrt(2), "B": 1 / np.sqrt(2)})
    qc = QuantumCognition()
    probs = qc.evaluate_probabilities(combined)
    assert probs["0"] == pytest.approx(0.5, abs=1e-6)
    assert probs["1"] == pytest.approx(0.5, abs=1e-6)


def test_superposition_partial_interference():
    memory = QuantumMemory()
    memory.store("A", SuperpositionState({"0": 1}))
    memory.store("B", SuperpositionState({"0": 1, "1": 1}))
    combined = memory.superposition({"A": 1 / np.sqrt(2), "B": 1 / np.sqrt(2)})
    qc = QuantumCognition()
    probs = qc.evaluate_probabilities(combined)
    assert probs["1"] == pytest.approx(0.1464466, abs=1e-6)
    assert probs["0"] == pytest.approx(0.8535534, abs=1e-6)
