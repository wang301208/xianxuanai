"""Tests for the generic genetic algorithm."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.getcwd()))

from evolution.generic_ga import GAConfig, GeneticAlgorithm  # noqa: E402


def sphere(x):
    return -sum(v * v for v in x)


def test_ga_optimizes_simple_function():
    bounds = [(-5, 5)]
    config = GAConfig(population_size=30, mutation_sigma=0.5)
    ga = GeneticAlgorithm(sphere, bounds, config)
    best, fitness = ga.run(generations=20)
    # optimum is at 0 with fitness 0
    assert abs(best[0]) < 1.0
    assert fitness > -1.0
