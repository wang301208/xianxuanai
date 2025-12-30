"""Gradient descent optimizer using energy minimization analogy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class GDResult:
    """Result of gradient descent optimization.

    Attributes:
        position: Final position found by the optimizer.
        value: Objective function value at ``position``.
    """

    position: np.ndarray
    value: float


def numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-8) -> np.ndarray:
    """Estimate the gradient of ``f`` at ``x`` using finite differences."""
    grad = np.zeros_like(x)
    fx = f(x)
    for i in range(len(x)):
        xh = x.copy()
        xh[i] += h
        grad[i] = (f(xh) - fx) / h
    return grad


def gradient_descent(
    f: Callable[[np.ndarray], float],
    start: Sequence[float],
    learning_rate: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> GDResult:
    """Minimise ``f`` using gradient descent.

    Args:
        f: Objective function representing an energy landscape.
        start: Initial position.
        learning_rate: Step size for descent.
        max_iter: Maximum number of iterations.
        tol: Convergence threshold on position change.

    Returns:
        ``GDResult`` with the best position and value found.
    """

    x = np.array(start, dtype=float)
    for _ in range(max_iter):
        grad = numerical_gradient(f, x)
        x_new = x - learning_rate * grad
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
    return GDResult(position=x, value=float(f(x)))
