"""Tests for StructuralEvolutionManager."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.evolution import (
    DynamicArchitectureExpander,
    EvolvingCognitiveArchitecture,
    EvolutionGeneticAlgorithm,
    SelfEvolvingAIArchitecture,
    StructuralEvolutionManager,
    StructuralGenome,
    ModuleGene,
    EdgeGene,
)
from modules.evolution.evolving_cognitive_architecture import GAConfig


def _fitness(arch):
    active = sum(1 for k, v in arch.items() if str(k).startswith("module_") and v >= 0.5)
    edges = sum(1 for k in arch.keys() if str(k).startswith("edge_"))
    return float(1.0 + active * 0.1 - edges * 0.05)


def test_structural_manager_disables_bottleneck_and_updates_topology():
    ga = EvolutionGeneticAlgorithm(_fitness, GAConfig(population_size=4, generations=1))
    evolver = EvolvingCognitiveArchitecture(_fitness, ga)
    architecture = SelfEvolvingAIArchitecture({"reward": 1.0}, evolver)

    expander = DynamicArchitectureExpander(
        modules={"a": lambda x: x + 1, "b": lambda x: x * 2},
        connections={"a": ["b"], "b": []},
        evolver=architecture,
    )

    manager = StructuralEvolutionManager(
        architecture=architecture,
        expander=expander,
        exploration_budget=2,
        connection_penalty=0.25,
    )

    proposal = manager.evolve_structure(
        performance=0.2,
        bottlenecks=[("b", 10.0)],
    )

    assert proposal.reason.startswith("disable")
    assert manager.module_gates["b"] == 0.0
    assert expander.connections["a"] == []
    assert architecture.architecture.get("module_b_active") == 0.0


def test_structural_manager_enables_idle_module_and_rewires():
    ga = EvolutionGeneticAlgorithm(_fitness, GAConfig(population_size=3, generations=1))
    evolver = EvolvingCognitiveArchitecture(_fitness, ga)
    architecture = SelfEvolvingAIArchitecture({"reward": 1.0}, evolver)

    expander = DynamicArchitectureExpander(
        modules={"a": lambda x: x + 1, "b": lambda x: x - 1, "c": lambda x: x * 3},
        connections={"a": ["b"], "b": [], "c": []},
        evolver=architecture,
    )

    manager = StructuralEvolutionManager(
        architecture=architecture,
        expander=expander,
        exploration_budget=3,
        connection_penalty=0.05,
        initial_module_gates={"c": 0.0},
    )

    proposal = manager.evolve_structure(
        performance=1.0,
        bottlenecks=[("a", 1.0)],
        candidate_modules=["c"],
    )

    assert proposal.reason.startswith("enable")
    assert manager.module_gates["c"] == 1.0
    assert any("c" in edges for edges in expander.connections.values())
    assert architecture.architecture.get("module_c_active") == 1.0


def test_structural_manager_accepts_genome_initial_state():
    genome = StructuralGenome(
        modules=[ModuleGene("a"), ModuleGene("b", enabled=False)],
        edges=[EdgeGene("a", "b", enabled=True)],
        next_innovation=2,
    )

    ga = EvolutionGeneticAlgorithm(_fitness, GAConfig(population_size=2, generations=1))
    evolver = EvolvingCognitiveArchitecture(_fitness, ga)
    architecture = SelfEvolvingAIArchitecture({"reward": 1.0}, evolver)

    manager = StructuralEvolutionManager(architecture=architecture, genome=genome)

    # Gates reflect genome enabled flags.
    assert manager.module_gates["a"] == 1.0
    assert manager.module_gates["b"] == 0.0
    # Edge skipped because dst is disabled.
    assert manager.topology["a"] == []
