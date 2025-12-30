import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml.ant_colony import ACOParameters, AntColony


def test_ant_colony_finds_optimal_tour():
    distance = np.array(
        [
            [0, 1, 3, 4],
            [1, 0, 1, 5],
            [3, 1, 0, 2],
            [4, 5, 2, 0],
        ],
        dtype=float,
    )
    params = ACOParameters(alpha=1, beta=5, rho=0.1, q=1.0, seed=42)
    colony = AntColony(distance, n_ants=10, n_iterations=50, params=params)
    path, cost = colony.run()

    assert path[0] == path[-1]
    assert len(set(path[:-1])) == distance.shape[0]
    assert cost <= 8.0 + 1e-6

