"""Tests for the NEAT-style neuroevolution backend."""

import random

from modules.evolution.dynamic_architecture import DynamicArchitectureExpander
from modules.evolution.evolving_cognitive_architecture import (
    EvolvingCognitiveArchitecture,
    GAConfig,
    GeneticAlgorithm,
)
from modules.evolution.neuroevolution_backend import (
    CognitiveNetworkGenome,
    NeuroevolutionBackend,
)
from modules.evolution.self_evolving_ai_architecture import SelfEvolvingAIArchitecture
from modules.evolution.structural_evolution import StructuralEvolutionManager


def test_neuro_backend_evolves_topology():
    base = CognitiveNetworkGenome.from_topology({"in": ["out"]}, {"in": 1.0, "out": 1.0})

    def _fitness(genome: CognitiveNetworkGenome) -> float:
        topology, _ = genome.to_topology()
        return float(len(topology.get("in", [])))

    backend = NeuroevolutionBackend(
        _fitness, base_genome=base, population_size=3, random_state=0
    )
    best, score = backend.evolve(generations=3)

    topology, gates = best.to_topology()
    assert isinstance(best, CognitiveNetworkGenome)
    assert score >= 0.0
    assert "out" in topology.get("in", [])
    assert all(value >= 0.0 for value in gates.values())


def test_architecture_applies_neuro_winner_into_history():
    random.seed(0)

    def simple_fitness(arch):
        return arch.get("edge_input->output", 0.0) + arch.get("module_input_active", 0.0)

    ga = GeneticAlgorithm(simple_fitness, GAConfig(population_size=4, generations=1))
    evolver = EvolvingCognitiveArchitecture(simple_fitness, ga)
    architecture = SelfEvolvingAIArchitecture({"weight": 1.0}, evolver)

    expander = DynamicArchitectureExpander(modules={"input": lambda x: x, "output": lambda x: x})
    base_genome = CognitiveNetworkGenome.from_topology(
        {"input": ["output"]}, {"input": 1.0, "output": 1.0}
    )
    manager = StructuralEvolutionManager(architecture, expander=expander, genome=base_genome.structural)
    architecture.attach_structural_manager(manager)
    architecture.neuro_backend = NeuroevolutionBackend(
        architecture._neuro_fitness, base_genome=base_genome, random_state=1, population_size=4
    )

    architecture.run_neuroevolution_cycle(generations=2, reason="test_neat_run")

    assert manager.history, "NEAT proposals should be recorded"
    assert architecture.history[-1].metrics.get("neuro_fitness") is not None
    assert expander.connections == manager.topology
    assert architecture.history[-1].architecture.get("structural_version") >= 1.0
