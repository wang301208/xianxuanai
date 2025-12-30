"""Evolutionary learning utilities for neuroevolution and policy search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple
import random


@dataclass
class GAConfig:
    population_size: int = 30
    tournament_size: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    mutation_sigma: float = 0.15


class _GeneticOptimizer:
    def __init__(self, fitness_fn: Callable[[Sequence[float]], float], bounds: Sequence[Tuple[float, float]], config: GAConfig) -> None:
        self.fitness_fn = fitness_fn
        self.bounds = list(bounds)
        self.config = config
        self.num_genes = len(bounds)
        self.best_vector: List[float] | None = None
        self.best_score: float | None = None

    def _random_vector(self) -> List[float]:
        return [random.uniform(low, high) for (low, high) in self.bounds]

    def _evaluate(self, population: List[List[float]]) -> List[float]:
        return [self.fitness_fn(ind) for ind in population]

    def _tournament(self, population: List[List[float]], fitnesses: List[float]) -> List[float]:
        idxs = random.sample(range(len(population)), self.config.tournament_size)
        best_idx = max(idxs, key=lambda idx: fitnesses[idx])
        return population[best_idx][:]

    def _crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        if random.random() >= self.config.crossover_rate or self.num_genes <= 1:
            return parent1[:], parent2[:]
        point = random.randrange(1, self.num_genes)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def _mutate(self, individual: List[float]) -> None:
        for i, (low, high) in enumerate(self.bounds):
            if random.random() < self.config.mutation_rate:
                individual[i] += random.gauss(0.0, self.config.mutation_sigma)
                individual[i] = max(low, min(high, individual[i]))

    def _update_best(self, population: List[List[float]], fitnesses: List[float]) -> None:
        best_idx = max(range(len(population)), key=lambda idx: fitnesses[idx])
        score = fitnesses[best_idx]
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_vector = population[best_idx][:]

    def run(self, generations: int) -> Tuple[List[float], float]:
        population = [self._random_vector() for _ in range(self.config.population_size)]
        fitnesses = self._evaluate(population)
        self._update_best(population, fitnesses)

        for _ in range(generations):
            new_population: List[List[float]] = []
            while len(new_population) < self.config.population_size:
                p1 = self._tournament(population, fitnesses)
                p2 = self._tournament(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                self._mutate(c1)
                self._mutate(c2)
                new_population.extend([c1, c2])
            population = new_population[: self.config.population_size]
            fitnesses = self._evaluate(population)
            self._update_best(population, fitnesses)

        assert self.best_vector is not None and self.best_score is not None
        return self.best_vector, self.best_score


class CandidateEvaluator(Protocol):
    """Protocol for evaluating a candidate parameter dictionary."""

    def __call__(self, parameters: Mapping[str, float]) -> float:
        ...


@dataclass(frozen=True)
class SearchDimension:
    """Describe a single hyper-parameter to optimise."""

    name: str
    low: float
    high: float
    as_int: bool = False


@dataclass
class EvolutionRecord:
    parameters: Dict[str, float]
    fitness: float


class EvolutionaryLearner:
    """Generic wrapper around a light-weight genetic algorithm for parameter evolution."""

    def __init__(
        self,
        search_space: Sequence[SearchDimension],
        evaluator: CandidateEvaluator,
        *,
        ga_config: GAConfig | None = None,
    ) -> None:
        if not search_space:
            raise ValueError("search_space must contain at least one dimension")
        self.search_space = list(search_space)
        self.evaluator = evaluator
        self.records: List[EvolutionRecord] = []
        bounds = [(dim.low, dim.high) for dim in self.search_space]

        def fitness(vector: Sequence[float]) -> float:
            params = self._vector_to_params(vector)
            score = float(self.evaluator(params))
            self.records.append(EvolutionRecord(params, score))
            return score

        self.ga = _GeneticOptimizer(fitness, bounds, ga_config or GAConfig())

    # ------------------------------------------------------------------ #
    def run(self, generations: int) -> Tuple[Dict[str, float], float]:
        best_vector, best_fitness = self.ga.run(generations)
        return self._vector_to_params(best_vector), float(best_fitness)

    # ------------------------------------------------------------------ #
    def _vector_to_params(self, vector: Sequence[float]) -> Dict[str, float]:
        params: Dict[str, float] = {}
        for value, dim in zip(vector, self.search_space):
            if dim.as_int:
                params[dim.name] = float(int(round(value)))
            else:
                params[dim.name] = float(value)
        return params


@dataclass
class PopulationResult:
    best_parameters: Dict[str, float]
    best_fitness: float
    history: List[EvolutionRecord]


class EvolutionaryPopulation:
    """Manage a full evolutionary run with optional evaluation hooks."""

    def __init__(
        self,
        search_space: Sequence[SearchDimension],
        evaluator: CandidateEvaluator,
        *,
        ga_config: GAConfig | None = None,
        callbacks: Optional[Iterable[Callable[[EvolutionRecord], None]]] = None,
    ) -> None:
        self.learner = EvolutionaryLearner(search_space, evaluator, ga_config=ga_config)
        self._callbacks = list(callbacks or [])

    def run(self, generations: int) -> PopulationResult:
        best_params, best_fitness = self.learner.run(generations)
        for record in self.learner.records:
            for callback in self._callbacks:
                callback(record)
        return PopulationResult(best_params, best_fitness, list(self.learner.records))
