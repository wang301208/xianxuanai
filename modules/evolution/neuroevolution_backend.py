"""Lightweight neuroevolution backend for cognitive network topologies."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .structural_genome import EdgeGene, ModuleGene, StructuralGenome


@dataclass
class CognitiveNodeGene:
    """Node-level gene for cognitive networks."""

    id: str
    bias: float = 0.0
    activation: str = "linear"
    enabled: bool = True


@dataclass
class CognitiveConnectionGene:
    """Connection-level gene mirroring NEAT-style edges."""

    src: str
    dst: str
    weight: float = 1.0
    enabled: bool = True


@dataclass
class CognitiveNetworkGenome:
    """Aggregate genome tying node/connection genes to a structural genome."""

    structural: StructuralGenome = field(default_factory=StructuralGenome)
    nodes: Dict[str, CognitiveNodeGene] = field(default_factory=dict)
    connections: List[CognitiveConnectionGene] = field(default_factory=list)
    metadata: Dict[str, float] = field(default_factory=dict)

    def clone(self) -> "CognitiveNetworkGenome":
        return CognitiveNetworkGenome(
            structural=self.structural.clone(),
            nodes={nid: CognitiveNodeGene(**vars(node)) for nid, node in self.nodes.items()},
            connections=[CognitiveConnectionGene(**vars(conn)) for conn in self.connections],
            metadata=dict(self.metadata),
        )

    def to_topology(self) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
        """Translate the genome to runtime topology and module gates."""

        topology, gates = self.structural.to_topology()
        for name, node in self.nodes.items():
            gates.setdefault(name, 1.0 if node.enabled else 0.0)
            topology.setdefault(name, [])
        for conn in self.connections:
            if not conn.enabled:
                continue
            if gates.get(conn.src, 0.0) < 0.5 or gates.get(conn.dst, 0.0) < 0.5:
                continue
            topology.setdefault(conn.src, [])
            if conn.dst not in topology[conn.src]:
                topology[conn.src].append(conn.dst)
        return topology, gates

    @classmethod
    def from_topology(
        cls,
        topology: Dict[str, List[str]],
        gates: Optional[Dict[str, float]] = None,
    ) -> "CognitiveNetworkGenome":
        gates = gates or {name: 1.0 for name in topology}
        modules = [
            ModuleGene(id=name, enabled=bool(value >= 0.5)) for name, value in gates.items()
        ]
        edges: List[EdgeGene] = []
        innovation = 0
        for src, dsts in topology.items():
            for dst in dsts:
                edges.append(EdgeGene(src=src, dst=dst, enabled=True, innovation=innovation))
                innovation += 1
        structural = StructuralGenome(modules, edges, next_innovation=innovation)
        return cls(structural=structural)


class NeuroevolutionBackend:
    """Minimal NEAT-inspired population manager."""

    def __init__(
        self,
        fitness_fn: Callable[[CognitiveNetworkGenome], float],
        *,
        base_genome: Optional[CognitiveNetworkGenome] = None,
        population_size: int = 6,
        mutation_rate: float = 0.3,
        random_state: Optional[int] = None,
    ) -> None:
        self.fitness_fn = fitness_fn
        self.population_size = max(2, population_size)
        self.mutation_rate = max(0.0, min(1.0, mutation_rate))
        self.random = random.Random(random_state)
        self.base_genome = base_genome or CognitiveNetworkGenome()
        self._population: List[CognitiveNetworkGenome] = [
            self._spawn_genome(i) for i in range(self.population_size)
        ]
        self.history: List[Tuple[float, CognitiveNetworkGenome]] = []

    def evolve(self, generations: int = 1) -> Tuple[CognitiveNetworkGenome, float]:
        """Run ``generations`` of evolution and return the fittest genome."""

        best: Optional[Tuple[CognitiveNetworkGenome, float]] = None
        for _ in range(max(1, generations)):
            scores = [(genome, float(self.fitness_fn(genome))) for genome in self._population]
            scores.sort(key=lambda item: item[1], reverse=True)
            self.history.append((scores[0][1], scores[0][0].clone()))
            best = scores[0]
            elites = [scores[0][0].clone(), scores[1][0].clone()]
            offspring: List[CognitiveNetworkGenome] = elites[:]
            while len(offspring) < self.population_size:
                parent = self.random.choice(elites)
                child = parent.clone()
                self._mutate(child)
                offspring.append(child)
            self._population = offspring
        if best is None:
            best = (self._population[0], float(self.fitness_fn(self._population[0])))
        return best[0], best[1]

    def _spawn_genome(self, index: int) -> CognitiveNetworkGenome:
        genome = self.base_genome.clone()
        genome.metadata["seed_index"] = float(index)
        self._mutate(genome)
        return genome

    def _mutate(self, genome: CognitiveNetworkGenome) -> None:
        if self.random.random() < self.mutation_rate:
            self._toggle_random_module(genome)
        if self.random.random() < self.mutation_rate:
            self._toggle_random_edge(genome)
        if self.random.random() < self.mutation_rate:
            self._add_random_edge(genome)

    def _toggle_random_module(self, genome: CognitiveNetworkGenome) -> None:
        all_modules = list(genome.structural.modules.values())
        if not all_modules:
            return
        target = self.random.choice(all_modules)
        genome.structural.toggle_module(target.id)

    def _toggle_random_edge(self, genome: CognitiveNetworkGenome) -> None:
        if not genome.structural.edges:
            return
        edge = self.random.choice(genome.structural.edges)
        genome.structural.toggle_edge(edge.src, edge.dst)

    def _add_random_edge(self, genome: CognitiveNetworkGenome) -> None:
        modules = list(genome.structural.modules.keys())
        if len(modules) < 2:
            return
        src, dst = self.random.sample(modules, 2)
        genome.structural.add_edge(src, dst, enabled=True)
