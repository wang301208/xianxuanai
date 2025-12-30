"""Benchmark demonstrating architecture evolution."""

from __future__ import annotations

import os
import random
import sys
from typing import Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.evolution.evolving_cognitive_architecture import (
    EvolvingCognitiveArchitecture,
)


def fitness(arch: Dict[str, float]) -> float:
    """Simple fitness: maximise the negative sum of squares."""
    return -(arch["x"] ** 2 + arch["y"] ** 2)


def run() -> float:
    random.seed(0)
    eca = EvolvingCognitiveArchitecture(fitness)
    arch = {"x": 1.0, "y": 1.0}
    perf = fitness(arch)
    for _ in range(5):
        arch = eca.evolve_architecture(arch, perf)
        perf = fitness(arch)
    return perf


if __name__ == "__main__":
    before = fitness({"x": 1.0, "y": 1.0})
    after = run()
    print(f"Initial performance: {before:.4f}")
    print(f"Evolved performance: {after:.4f}")
