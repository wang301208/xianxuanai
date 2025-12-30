"""Simple experiment to measure the benefit of exploration strategies.

The script runs the :class:`modules.evolution.generic_ga.GeneticAlgorithm` on a
one-dimensional multimodal function.  It compares success rates and average
performance with and without the :class:`~modules.evolution.strategy.SimulatedAnnealingStrategy`.
"""

from __future__ import annotations

import math
import os
import sys
from statistics import mean
from typing import Tuple

# Ensure repository root on path for direct execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.evolution.generic_ga import GeneticAlgorithm
from modules.evolution.strategy import SimulatedAnnealingStrategy


# Target function with many local optima

def objective(ind: Tuple[float, ...]) -> float:
    x = ind[0]
    return math.sin(5 * x) * (1 - math.tanh(x * x))


BOUNDS = [(-2.0, 2.0)]
GENERATIONS = 40
TRIALS = 20
SUCCESS_THRESHOLD = 0.9


def run_trial(use_strategy: bool) -> float:
    strategy = SimulatedAnnealingStrategy() if use_strategy else None
    ga = GeneticAlgorithm(objective, bounds=BOUNDS, strategy=strategy)
    _, best = ga.run(GENERATIONS)
    return best


def run_experiment() -> None:
    for use_strategy in (False, True):
        results = [run_trial(use_strategy) for _ in range(TRIALS)]
        success_rate = sum(r > SUCCESS_THRESHOLD for r in results) / TRIALS
        avg_score = mean(results)
        label = "with" if use_strategy else "without"
        print(
            f"Simulated annealing {label} strategy: "
            f"success rate={success_rate:.2f}, avg best={avg_score:.3f}"
        )


if __name__ == "__main__":
    run_experiment()
