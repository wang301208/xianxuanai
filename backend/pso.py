"""Basic Particle Swarm Optimization (PSO) implementation.

This module provides a simple implementation of the Particle Swarm
Optimization algorithm suitable for educational purposes and small
optimization tasks. The algorithm follows the canonical PSO update rules
with inertia weight and supports basic strategies to prevent particle
stagnation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Tuple, List, Optional

import logging

import numpy as np


@dataclass
class PSOResult:
    """Result of a PSO run."""

    position: np.ndarray
    value: float
    w_history: List[float] = field(default_factory=list)
    c1_history: List[float] = field(default_factory=list)
    c2_history: List[float] = field(default_factory=list)


def linear_schedule(start: float, end: float, max_iter: int) -> Callable[[int, float], float]:
    """Create a linear schedule function.

    The returned callable maps ``iteration`` and the current value to the
    coefficient linearly interpolated between ``start`` and ``end``.
    """

    def schedule(iteration: int, _: float) -> float:
        return start + (end - start) * (iteration / max_iter)

    return schedule


def pso(
    f: Callable[[np.ndarray], float],
    bounds: Iterable[Tuple[float, float]],
    num_particles: int = 30,
    max_iter: int = 100,
    w: float = 0.9,
    c1: float = 2.0,
    c2: float = 2.0,
    w_schedule: Optional[Callable[[int, float], float]] = None,
    c1_schedule: Optional[Callable[[int, float], float]] = None,
    c2_schedule: Optional[Callable[[int, float], float]] = None,
    convergence_patience: int = 20,
    convergence_tol: float = 1e-6,
    convergence_adjust: Tuple[float, float, float] = (0.9, 0.9, 0.9),
    log_params: bool = False,
) -> PSOResult:
    """Run Particle Swarm Optimization on ``f``.

    Args:
        f: Objective function to minimise. It must accept a NumPy array and
            return a scalar.
        bounds: Iterable of ``(lower, upper)`` tuples defining the search
            space for each dimension.
        num_particles: Number of particles in the swarm.
        max_iter: Maximum number of iterations to perform.
        w: Inertia weight controlling impact of previous velocity.
        c1: Cognitive acceleration coefficient.
        c2: Social acceleration coefficient.

    Returns:
        ``PSOResult`` containing the best position and its fitness value.
    """

    logger = logging.getLogger(__name__)

    if w_schedule is None:
        w_schedule = lambda _, val: 0.99 * val
    if c1_schedule is None:
        c1_schedule = lambda _, val: val
    if c2_schedule is None:
        c2_schedule = lambda _, val: val

    bounds = np.array(list(bounds), dtype=float)
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    dim = len(bounds)

    rng = np.random.default_rng()
    x = rng.uniform(lower, upper, size=(num_particles, dim))
    v = np.zeros_like(x)

    # Evaluate initial population
    p_best = x.copy()
    p_best_val = np.apply_along_axis(f, 1, x)
    best_idx = np.argmin(p_best_val)
    g_best = p_best[best_idx].copy()
    g_best_val = p_best_val[best_idx]

    w_hist: List[float] = []
    c1_hist: List[float] = []
    c2_hist: List[float] = []
    no_improve = 0
    prev_best = g_best_val

    for it in range(max_iter):
        w = w_schedule(it, w)
        c1 = c1_schedule(it, c1)
        c2 = c2_schedule(it, c2)

        r1 = rng.random((num_particles, dim))
        r2 = rng.random((num_particles, dim))

        v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)
        x = x + v

        # Boundary handling (clamp)
        x = np.clip(x, lower, upper)

        # Evaluate and update bests
        values = np.apply_along_axis(f, 1, x)
        improved = values < p_best_val
        p_best[improved] = x[improved]
        p_best_val[improved] = values[improved]

        best_idx = np.argmin(p_best_val)
        if p_best_val[best_idx] < g_best_val:
            g_best_val = p_best_val[best_idx]
            g_best = p_best[best_idx].copy()

        if g_best_val < prev_best - convergence_tol:
            no_improve = 0
            prev_best = g_best_val
        else:
            no_improve += 1
            if no_improve >= convergence_patience:
                w *= convergence_adjust[0]
                c1 *= convergence_adjust[1]
                c2 *= convergence_adjust[2]
                no_improve = 0

        # Re-randomise stagnant particles
        stagnation = np.linalg.norm(v, axis=1) < 1e-5
        if np.any(stagnation):
            x[stagnation] = rng.uniform(lower, upper, size=(stagnation.sum(), dim))
            v[stagnation] = 0

        if log_params:
            w_hist.append(w)
            c1_hist.append(c1)
            c2_hist.append(c2)
            logger.debug("iter %d: w=%.4f c1=%.4f c2=%.4f", it, w, c1, c2)

    return PSOResult(
        position=g_best,
        value=float(g_best_val),
        w_history=w_hist,
        c1_history=c1_hist,
        c2_history=c2_hist,
    )
