"""Tests for DynamicArchitectureExpander."""

import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.evolution import (
    DynamicArchitectureExpander,
    EvolvingCognitiveArchitecture,
    EvolutionGeneticAlgorithm,
    SelfEvolvingAIArchitecture,
)
from modules.evolution.evolving_cognitive_architecture import GAConfig


def fitness_fn(arch):
    return float(len(arch))


def test_expander_adds_module_and_produces_new_behavior():
    random.seed(0)
    ga = EvolutionGeneticAlgorithm(fitness_fn, GAConfig(population_size=5, generations=2))
    evolver = EvolvingCognitiveArchitecture(fitness_fn, ga)
    self_arch = SelfEvolvingAIArchitecture({"inc": 1.0}, evolver)

    expander = DynamicArchitectureExpander(
        modules={"inc": lambda x: x + 1},
        connections={"inc": []},
        evolver=self_arch,
    )

    # Original behaviour: only increment.
    assert expander.run("inc", 3) == 4

    # Add doubling module via environment feedback and trigger auto expansion.
    expander.auto_expand(
        performance=0.1,
        env_feedback={"add_module": ("dbl", lambda x: x * 2, "inc")},
    )

    # The architecture is updated in the evolver.
    assert "dbl" in self_arch.architecture

    # New behaviour should increment then double.
    assert expander.run("inc", 3) == 8
