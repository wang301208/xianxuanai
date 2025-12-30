"""Common benchmark optimization problems with known optima."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple
import math
import numpy as np


@dataclass
class Problem:
    """Base class for optimization problems."""

    dim: int
    bounds: Sequence[Tuple[float, float]]
    optimum: Sequence[float]
    optimum_value: float
    name: str

    def evaluate(self, x: Sequence[float]) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def clip(self, x: np.ndarray) -> np.ndarray:
        """Clip array ``x`` to the problem bounds."""
        lower = np.array([b[0] for b in self.bounds])
        upper = np.array([b[1] for b in self.bounds])
        return np.minimum(np.maximum(x, lower), upper)


class Sphere(Problem):
    """Sphere function ``f(x) = sum(x_i^2)`` with optimum at ``x=0``."""

    def __init__(self, dim: int = 2, bound: float = 5.12):
        bounds = [(-bound, bound)] * dim
        super().__init__(
            dim=dim,
            bounds=bounds,
            optimum=[0.0] * dim,
            optimum_value=0.0,
            name="sphere",
        )

    def evaluate(self, x: Sequence[float]) -> float:
        return float(np.sum(np.square(x)))


class Rastrigin(Problem):
    """Rastrigin function with global minimum at ``x=0``."""

    def __init__(self, dim: int = 2, bound: float = 5.12):
        bounds = [(-bound, bound)] * dim
        super().__init__(
            dim=dim,
            bounds=bounds,
            optimum=[0.0] * dim,
            optimum_value=0.0,
            name="rastrigin",
        )

    def evaluate(self, x: Sequence[float]) -> float:
        x = np.asarray(x)
        return float(10 * self.dim + np.sum(x * x - 10 * np.cos(2 * math.pi * x)))


class ConstrainedQuadratic(Problem):
    """Minimise ``x^2 + y^2`` subject to ``x + y = 1``."""

    def __init__(self):
        bounds = [(0.0, 1.0)] * 2
        super().__init__(
            dim=2,
            bounds=bounds,
            optimum=[0.5, 0.5],
            optimum_value=0.5,
            name="constrained_quadratic",
        )

    def is_feasible(self, x: Sequence[float]) -> bool:
        return abs(sum(x) - 1.0) < 1e-6

    def evaluate(self, x: Sequence[float]) -> float:
        if not self.is_feasible(x):
            return float("inf")
        x = np.asarray(x)
        return float(np.sum(np.square(x)))
