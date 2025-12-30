from __future__ import annotations

import pickle
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Sequence, Tuple

from .ga_config import GAConfig


class ParallelGA:
    """Simple parallel genetic algorithm implementation."""

    def __init__(self, fitness_fn: Callable[[Sequence[float]], float], config: GAConfig):
        self.fitness_fn = fitness_fn
        self.config = config
        self.cache: dict[Tuple[float, ...], float] = {}
        if self.config.cache_path.exists():
            try:
                with open(self.config.cache_path, "rb") as f:
                    self.cache = pickle.load(f)
            except Exception:
                self.cache = {}

    # ------------------------------------------------------------------
    def _save_cache(self) -> None:
        try:
            with open(self.config.cache_path, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception:
            pass

    def _random_individual(self) -> List[float]:
        return [random.random() for _ in range(self.config.gene_length)]

    def _evaluate(self, individual: Sequence[float]) -> float:
        key = tuple(individual)
        if key in self.cache:
            return self.cache[key]
        fitness = self.fitness_fn(individual)
        self.cache[key] = fitness
        return fitness

    def _evaluate_population(self, population: List[Sequence[float]]) -> List[float]:
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            fitnesses = list(executor.map(self._evaluate, population))
        self._save_cache()
        return fitnesses

    # Genetic operators -------------------------------------------------
    def _select_parents(self, population: List[Sequence[float]], fitnesses: List[float]) -> List[Sequence[float]]:
        parents: List[Sequence[float]] = []
        for _ in range(len(population)):
            i, j = random.sample(range(len(population)), 2)
            winner = population[i] if fitnesses[i] > fitnesses[j] else population[j]
            parents.append(winner)
        return parents

    def _crossover(self, p1: Sequence[float], p2: Sequence[float]) -> List[float]:
        if random.random() > self.config.crossover_rate:
            return list(p1)
        point = random.randint(1, self.config.gene_length - 1)
        return list(p1[:point] + p2[point:])

    def _mutate(self, individual: List[float]) -> List[float]:
        for i in range(len(individual)):
            if random.random() < self.config.mutation_rate:
                individual[i] = random.random()
        return individual

    # Public API --------------------------------------------------------
    def run(self) -> Tuple[List[float], float, List[dict[str, float]]]:
        population = [self._random_individual() for _ in range(self.config.population_size)]
        best: List[float] | None = None
        best_fit = float("-inf")
        start = time.time()
        no_improve = 0
        history: List[dict[str, float]] = []

        for generation in range(1, self.config.max_generations + 1):
            fitnesses = self._evaluate_population(population)
            gen_best_fit = max(fitnesses)
            gen_best = population[fitnesses.index(gen_best_fit)]
            elapsed = time.time() - start
            history.append({"generation": generation, "elapsed_time": elapsed, "best_fitness": gen_best_fit})

            if gen_best_fit > best_fit:
                best_fit = gen_best_fit
                best = list(gen_best)
                no_improve = 0
            else:
                no_improve += 1

            if (
                self.config.target_fitness is not None and best_fit >= self.config.target_fitness
            ) or no_improve >= self.config.early_stopping_rounds:
                break

            parents = self._select_parents(population, fitnesses)
            next_population: List[List[float]] = []
            for i in range(0, len(parents), 2):
                p1 = parents[i]
                p2 = parents[(i + 1) % len(parents)]
                child1 = self._mutate(self._crossover(p1, p2))
                child2 = self._mutate(self._crossover(p2, p1))
                next_population.extend([child1, child2])
            population = next_population[: self.config.population_size]

        return best or [], best_fit, history
