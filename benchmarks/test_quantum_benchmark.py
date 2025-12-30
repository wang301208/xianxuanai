import os
import sys
import tracemalloc

import numpy as np

# Ensure repo root in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.brain.quantum.quantum_cognition import (
    EntanglementNetwork,
    QuantumCognition,
    SuperpositionState,
)
from modules.brain.quantum.quantum_memory import QuantumMemory


def test_make_decision_benchmark(benchmark):
    qc = QuantumCognition()
    options = {"A": [0.5, 0.5], "B": [0.5, -0.5]}
    benchmark(qc.make_decision, options)


def test_memory_superposition_benchmark(benchmark):
    memory = QuantumMemory()
    memory.store("A", SuperpositionState({"0": 1}))
    memory.store("B", SuperpositionState({"1": 1}))
    weights = {"A": 1 / np.sqrt(2), "B": 1 / np.sqrt(2)}
    benchmark(memory.superposition, weights)


def test_entangle_memory_usage():
    network = EntanglementNetwork()
    tracemalloc.start()
    network.entangle_concepts("cat", "hat", decoherence=0.3)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 10000
