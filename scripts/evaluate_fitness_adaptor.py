"""Compare manual and adaptive fitness functions within a genetic algorithm."""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure the repository root is on ``sys.path`` when executed directly.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from modules.evolution.generic_ga import GAConfig, GeneticAlgorithm
from modules.evolution.fitness_adaptor import AdaptiveFitnessGenerator


def metric_performance(individual):
    """Favor individuals where the first gene is close to 1."""
    return -abs(individual[0] - 1.0)


def metric_resource(individual):
    """Favor individuals where the second gene is close to -1."""
    return -abs(individual[1] + 1.0)


def metric_ethics(individual):
    """Favor individuals where the sum of genes stays near 0."""
    return -abs(individual[0] + individual[1])


def run_manual():
    bounds = [(-5, 5), (-5, 5)]
    metrics = [
        (metric_performance, 0.5),
        (metric_resource, 0.3),
        (metric_ethics, 0.2),
    ]
    ga = GeneticAlgorithm(bounds=bounds, metrics=metrics, config=GAConfig(population_size=30, mutation_sigma=0.5))
    return ga.run(generations=30)


def run_adaptive():
    bounds = [(-5, 5), (-5, 5)]
    metrics = [
        ("performance", metric_performance),
        ("resource", metric_resource),
        ("ethics", metric_ethics),
    ]
    adaptor = AdaptiveFitnessGenerator(metrics)

    step = 0

    def fitness_fn(individual):
        nonlocal step
        if step < 10:
            env = {"performance": 1.0, "resource": 0.5, "ethics": 0.2}
        elif step < 20:
            env = {"performance": 0.3, "resource": 1.0, "ethics": 0.2}
        else:
            env = {"performance": 0.3, "resource": 0.4, "ethics": 1.0}
        step += 1
        return adaptor(individual, env)

    ga = GeneticAlgorithm(bounds=bounds, fitness_fn=fitness_fn, config=GAConfig(population_size=30, mutation_sigma=0.5))
    return ga.run(generations=30)


def main() -> None:
    m_best, m_fit = run_manual()
    a_best, a_fit = run_adaptive()
    print("Manual best:", m_best, "fitness:", m_fit)
    print("Adaptive best:", a_best, "fitness:", a_fit)


if __name__ == "__main__":
    main()
