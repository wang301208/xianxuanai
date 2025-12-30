from __future__ import annotations

"""Exploration strategies for evolutionary algorithms.

This module provides simple operators that can be plugged into the
:class:`~modules.evolution.generic_ga.GeneticAlgorithm` to encourage search
beyond local optima.  Two strategies are implemented:

* :class:`SimulatedAnnealingStrategy` - occasionally accepts worse individuals
  with a probability governed by an annealing temperature.
* :class:`InnovationProtectionStrategy` - adds random noise to individuals that
  have been seen before to maintain population diversity.
"""

from dataclasses import dataclass, field
import math
import random
from typing import List, Tuple


class ExplorationStrategy:
    """Interface for exploration strategies."""

    def apply(self, population: List[List[float]], fitnesses: List[float], generation: int) -> None:
        """Modify ``population`` or ``fitnesses`` in-place.

        Parameters
        ----------
        population: List[List[float]]
            Current population of individuals.
        fitnesses: List[float]
            Fitness for each individual.  The list is mutated when fitnesses are
            adjusted.
        generation: int
            Index of the current generation.
        """
        raise NotImplementedError


@dataclass
class SimulatedAnnealingStrategy(ExplorationStrategy):
    """Accept worse individuals with a probability that decreases over time."""

    initial_temp: float = 1.0
    cooling_rate: float = 0.95

    def apply(self, population: List[List[float]], fitnesses: List[float], generation: int) -> None:
        temp = self.initial_temp * (self.cooling_rate ** generation)
        if temp <= 0:
            return
        best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        best_fit = fitnesses[best_idx]
        for i in range(len(population)):
            if fitnesses[i] < best_fit:
                prob = math.exp((fitnesses[i] - best_fit) / max(temp, 1e-12))
                if random.random() < prob:
                    population[best_idx], population[i] = population[i], population[best_idx]
                    fitnesses[best_idx], fitnesses[i] = fitnesses[i], fitnesses[best_idx]
                    best_idx = i
                    best_fit = fitnesses[i]


@dataclass
class InnovationProtectionStrategy(ExplorationStrategy):
    """Perturb duplicate individuals to protect novel solutions."""

    noise_scale: float = 0.1
    archive: set[Tuple[float, ...]] = field(default_factory=set)

    def apply(self, population: List[List[float]], fitnesses: List[float], generation: int) -> None:
        for ind in population:
            key = tuple(ind)
            if key in self.archive:
                for j in range(len(ind)):
                    ind[j] += random.uniform(-self.noise_scale, self.noise_scale)
            self.archive.add(tuple(ind))
