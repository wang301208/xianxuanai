import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from energy_minimization import gradient_descent  # noqa: E402


def quadratic_energy(x: np.ndarray) -> float:
    """Energy function with minimum at x=3 for each dimension."""
    return float(np.sum((x - 3) ** 2))


def test_gradient_descent_minimises_energy():
    start = np.array([10.0, -5.0])
    result = gradient_descent(quadratic_energy, start, learning_rate=0.1, max_iter=500)
    assert result.value < 1e-4
    assert np.allclose(result.position, np.array([3.0, 3.0]), atol=1e-2)
