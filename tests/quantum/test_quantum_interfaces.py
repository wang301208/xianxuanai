import os
import sys

import numpy as np
import pytest

# Ensure repo root in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.quantum.quantum_cognition import QuantumCognition, SuperpositionState


def test_quantum_memory_retrieval_superposition():
    qc = QuantumCognition()
    qc.memory.store("A", SuperpositionState({"0": 1}))
    qc.memory.store("B", SuperpositionState({"1": 1}))
    state = qc.quantum_memory_retrieval({"A": 1 / np.sqrt(2), "B": 1 / np.sqrt(2)})
    probs = qc.evaluate_probabilities(state)
    assert probs["0"] == pytest.approx(0.5, rel=1e-6)
    assert probs["1"] == pytest.approx(0.5, rel=1e-6)


def test_quantum_decision_making_interference():
    qc = QuantumCognition()
    choice, probs = qc.quantum_decision_making(
        {"A": [0.5, 0.5], "B": [0.5, -0.5]}, rng=np.random.default_rng(0)
    )
    assert choice == "A"
    assert probs["A"] == pytest.approx(1.0, abs=1e-6)
    assert probs["B"] == pytest.approx(0.0, abs=1e-6)
