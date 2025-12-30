import os
import sys

import numpy as np
import pytest

# Ensure repo root in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.quantum.quantum_cognition import QuantumCognition


def test_make_decision_deterministic():
    qc = QuantumCognition()
    options = {"A": [0.5, 0.5], "B": [0.5, -0.5]}
    choice, probs = qc.make_decision(options, rng=np.random.default_rng(0))
    assert choice == "A"
    assert probs["A"] == pytest.approx(1.0, abs=1e-6)
    assert probs["B"] == pytest.approx(0.0, abs=1e-6)


def test_make_decision_normalizes_amplitudes():
    qc = QuantumCognition()
    options = {"A": [1, 1], "B": [1, -1]}
    _, probs = qc.make_decision(options, rng=np.random.default_rng(1))
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)


def test_make_decision_statistical_distribution():
    qc = QuantumCognition()
    options = {"A": [1], "B": [1]}
    rng = np.random.default_rng(42)
    counts = {"A": 0, "B": 0}
    for _ in range(1000):
        choice, _ = qc.make_decision(options, rng)
        counts[choice] += 1
    freq_a = counts["A"] / 1000
    assert freq_a == pytest.approx(0.5, abs=0.05)
