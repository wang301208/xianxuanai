"""A tiny Genetic Algorithm implementation with a unified interface."""
from __future__ import annotations

from typing import Optional, Tuple
import time
import numpy as np

from .termination import StopCondition

from benchmarks.problems import Problem


def optimize(
    problem: Problem,
    seed: Optional[int] = None,
    max_iters: int = 100,
    max_time: Optional[float] = None,
    patience: Optional[int] = None,
    pop_size: int = 20,
    mutation_rate: float = 0.1,
) -> Tuple[np.ndarray, float, int, float]:
    """Optimize ``problem`` using a very small GA."""
    rng = np.random.default_rng(seed)

    lower = np.array([b[0] for b in problem.bounds])
    upper = np.array([b[1] for b in problem.bounds])

    pop = rng.uniform(lower, upper, size=(pop_size, problem.dim))
    fitness = np.array([problem.evaluate(ind) for ind in pop])

    best_idx = np.argmin(fitness)
    best, best_val = pop[best_idx], fitness[best_idx]

    stopper = StopCondition(max_iters=max_iters, max_time=max_time, patience=patience)
    while stopper.keep_running():
        # Select the best half
        idx = np.argsort(fitness)
        parents = pop[idx][: pop_size // 2]

        # Simple arithmetic crossover to create children
        children = []
        for i in range(0, len(parents), 2):
            p1 = parents[i]
            p2 = parents[(i + 1) % len(parents)]
            children.append((p1 + p2) / 2.0)
        children = np.array(children)

        pop = np.vstack([parents, children])
        if pop.shape[0] < pop_size:  # maintain population size
            extra = parents[: pop_size - pop.shape[0]]
            pop = np.vstack([pop, extra])

        # Gaussian mutation and clipping to bounds
        pop += rng.normal(0.0, mutation_rate, pop.shape)
        pop = problem.clip(pop)

        fitness = np.array([problem.evaluate(ind) for ind in pop])
        idx = np.argmin(fitness)
        improved = False
        if fitness[idx] < best_val:
            best_val = fitness[idx]
            best = pop[idx]
            improved = True
        stopper.update(improved)

    elapsed = time.time() - stopper.start_time
    return best, best_val, stopper.iteration, elapsed
