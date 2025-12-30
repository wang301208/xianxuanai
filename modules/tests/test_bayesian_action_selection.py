"""Tests for Bayesian action selection with entropy minimisation."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml.bayesian_action_selection import (
    BayesianActionSelector,
    BetaDistribution,
    simulate_strategies,
)


def test_posterior_update():
    dist = BetaDistribution(1.0, 1.0)
    dist.update(1)
    assert dist.alpha == 2.0
    assert dist.beta == 1.0
    dist.update(0)
    assert dist.alpha == 2.0
    assert dist.beta == 2.0


def test_information_gain_reduces_entropy_vs_random():
    true_probs = {"a": 0.7, "b": 0.4}
    entropies = simulate_strategies(true_probs, steps=50)
    assert entropies["information_gain"] < entropies["random"]
