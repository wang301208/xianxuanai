"""Baseline random search optimizer."""
from __future__ import annotations

from typing import Optional, Tuple
import time
import numpy as np

from .termination import StopCondition

from benchmarks.problems import Problem


def optimize(
    problem: Problem,
    seed: Optional[int] = None,
    max_iters: int = 1000,
    max_time: Optional[float] = None,
    patience: Optional[int] = None,
) -> Tuple[np.ndarray, float, int, float]:
    rng = np.random.default_rng(seed)
    lower = np.array([b[0] for b in problem.bounds])
    upper = np.array([b[1] for b in problem.bounds])
    best = None
    best_val = float("inf")
    stopper = StopCondition(max_iters=max_iters, max_time=max_time, patience=patience)
    while stopper.keep_running():
        x = rng.uniform(lower, upper)
        val = problem.evaluate(x)
        improved = False
        if val < best_val:
            best_val = val
            best = x
            improved = True
        stopper.update(improved)
    elapsed = time.time() - stopper.start_time
    return best, best_val, stopper.iteration, elapsed
