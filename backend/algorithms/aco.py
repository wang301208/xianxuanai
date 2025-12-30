"""A very small Ant Colony Optimization style optimizer."""
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
    ant_count: int = 20,
    q: float = 0.1,
) -> Tuple[np.ndarray, float, int, float]:
    rng = np.random.default_rng(seed)

    mean = np.array([(b[0] + b[1]) / 2 for b in problem.bounds])
    std = np.array([(b[1] - b[0]) / 2 for b in problem.bounds])

    best = None
    best_val = float("inf")

    stopper = StopCondition(max_iters=max_iters, max_time=max_time, patience=patience)
    while stopper.keep_running():
        ants = rng.normal(mean, std, size=(ant_count, problem.dim))
        ants = problem.clip(ants)
        values = np.array([problem.evaluate(a) for a in ants])
        idx = np.argmin(values)
        improved = False
        if values[idx] < best_val:
            best_val = values[idx]
            best = ants[idx]
            improved = True
        # update pheromone (mean) towards best ant
        mean = (1 - q) * mean + q * best
        std *= 0.95  # slowly reduce exploration
        stopper.update(improved)

    elapsed = time.time() - stopper.start_time
    return best, best_val, stopper.iteration, elapsed
