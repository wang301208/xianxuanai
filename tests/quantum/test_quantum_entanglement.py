import os
import sys

import numpy as np
import pytest

# Ensure repo root in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.quantum.quantum_cognition import EntanglementNetwork, QuantumCognition


def test_create_bell_pair_probabilities():
    network = EntanglementNetwork()
    bell = network.create_bell_pair()
    qc = QuantumCognition()
    probs = qc.evaluate_probabilities(bell)
    assert probs["00"] == pytest.approx(0.5, abs=1e-6)
    assert probs["11"] == pytest.approx(0.5, abs=1e-6)
    assert probs["01"] == pytest.approx(0.0, abs=1e-6)
    assert probs["10"] == pytest.approx(0.0, abs=1e-6)


def test_entangle_concepts_decoherence():
    network = EntanglementNetwork()
    density = network.entangle_concepts("cat", "hat", decoherence=0.3)
    qc = QuantumCognition()
    probs = qc.evaluate_probabilities(density)
    assert probs["00"] == pytest.approx(0.5, abs=1e-6)
    assert probs["11"] == pytest.approx(0.5, abs=1e-6)
    assert density[0, 3] == pytest.approx(0.35, rel=1e-6)
    assert density[3, 0] == pytest.approx(0.35, rel=1e-6)


def test_entangle_concepts_full_decoherence():
    network = EntanglementNetwork()
    density = network.entangle_concepts("cat", "hat", decoherence=1.0)
    assert density[0, 3] == pytest.approx(0.0, abs=1e-6)
    assert density[3, 0] == pytest.approx(0.0, abs=1e-6)
