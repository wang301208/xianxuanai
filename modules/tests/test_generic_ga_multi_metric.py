"""Tests for multi-metric fitness evaluation in the genetic algorithm."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.getcwd()))

from evolution.generic_ga import GAConfig, GeneticAlgorithm  # noqa: E402
from evolution import fitness_plugins  # noqa: E402


def test_ga_optimizes_multiple_metrics(tmp_path: Path) -> None:
    """Genetic algorithm should handle multiple weighted metrics."""

    cfg = tmp_path / "metrics.yaml"
    cfg.write_text(
        """
metrics:
  - name: minimize_response_time
    weight: 0.6
  - name: minimize_resource_consumption
    weight: 0.4
        """
    )

    metrics = fitness_plugins.load_from_config(cfg)
    bounds = [(-5, 5), (-5, 5)]
    config = GAConfig(population_size=30, mutation_sigma=0.5)
    ga = GeneticAlgorithm(bounds=bounds, metrics=metrics, config=config)
    best, _ = ga.run(generations=20)
    # Both genes should approach zero due to minimization in templates
    assert abs(best[0]) < 1.0
    assert abs(best[1]) < 1.0


def test_weighted_sum_evaluation(tmp_path: Path) -> None:
    cfg = tmp_path / "metrics.yaml"
    cfg.write_text(
        """
metrics:
  - name: minimize_response_time
    weight: 0.5
  - name: minimize_resource_consumption
    weight: 0.5
        """
    )
    metrics = fitness_plugins.load_from_config(cfg)
    ga = GeneticAlgorithm(bounds=[(-5, 5), (-5, 5)], metrics=metrics)
    sample = [2.0, -3.0]
    expected = sum(weight * fn(sample) for fn, weight in metrics)
    fitness = ga._evaluate([sample])[0]
    assert fitness == expected
