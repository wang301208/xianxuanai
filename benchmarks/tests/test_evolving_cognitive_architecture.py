"""Benchmark tests for EvolvingCognitiveArchitecture."""

from __future__ import annotations

import random
from typing import Dict

from modules.evolution.evolving_cognitive_architecture import (
    EvolvingCognitiveArchitecture,
)


def fitness(arch: Dict[str, float]) -> float:
    return -(arch["x"] ** 2 + arch["y"] ** 2)


def run_evolution() -> float:
    random.seed(0)
    eca = EvolvingCognitiveArchitecture(fitness)
    arch = {"x": 1.0, "y": 1.0}
    perf = fitness(arch)
    for _ in range(5):
        arch = eca.evolve_architecture(arch, perf)
        perf = fitness(arch)
    return perf


def test_evolution_improves_performance(benchmark):
    initial = fitness({"x": 1.0, "y": 1.0})
    final = benchmark(run_evolution)
    assert final > initial
