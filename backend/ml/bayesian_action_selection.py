"""Bayesian action selection with entropy-based exploration.

This module models uncertainty using Beta priors and updates the posterior
with Bernoulli observations. Actions are chosen by maximizing the expected
information gain, approximated by the variance of the Beta posterior.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping


def _digamma(x: float) -> float:
    """Approximate the digamma function for positive ``x``.

    The implementation uses an asymptotic expansion after increasing ``x`` to
    a sufficiently large value. The approximation is accurate enough for the
    small ``x`` values encountered in the simulations.
    """

    result = 0.0
    while x < 7:  # reduce to larger argument for better accuracy
        result -= 1.0 / x
        x += 1.0

    x -= 0.5
    inv = 1.0 / x
    inv2 = inv * inv
    return result + math.log(x) - inv / 2.0 - inv2 / 12.0 + inv2 * inv2 / 120.0


def _beta_entropy(alpha: float, beta: float) -> float:
    """Return the differential entropy of a Beta distribution."""

    log_beta = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    return (
        log_beta
        - (alpha - 1.0) * _digamma(alpha)
        - (beta - 1.0) * _digamma(beta)
        + (alpha + beta - 2.0) * _digamma(alpha + beta)
    )


@dataclass
class BetaDistribution:
    """Beta prior/posterior for a Bernoulli random variable."""

    alpha: float
    beta: float

    def update(self, observation: int) -> None:
        """Update with a Bernoulli observation ``0`` or ``1``."""

        self.alpha += observation
        self.beta += 1 - observation

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        a, b = self.alpha, self.beta
        n = a + b
        return a * b / (n * n * (n + 1.0))

    def entropy(self) -> float:
        return _beta_entropy(self.alpha, self.beta)


class BayesianActionSelector:
    """Select actions by maximizing expected information gain.

    Each action has an associated Beta prior. The ``select_action`` method
    chooses the action whose Beta posterior has the highest variance, which is
    equivalent to maximising the expected information gained from observing its
    outcome.
    """

    def __init__(self, priors: Mapping[str, Iterable[float]]):
        self.beliefs: Dict[str, BetaDistribution] = {
            name: BetaDistribution(alpha, beta) for name, (alpha, beta) in priors.items()
        }

    def select_action(self) -> str:
        return max(self.beliefs.items(), key=lambda item: item[1].variance())[0]

    def update(self, action: str, outcome: int) -> None:
        self.beliefs[action].update(outcome)


def simulate_strategies(
    true_success_probs: Mapping[str, float],
    steps: int,
    seed: int = 0,
) -> Dict[str, float]:
    """Validate the information gain strategy against a random baseline.

    ``true_success_probs`` maps action names to the true success probability
    used to sample Bernoulli outcomes. The returned dictionary contains the
    final total entropy for the information gain strategy and a random baseline
    heuristic after ``steps`` iterations of simulated interaction.
    """

    def run(select_fn: Callable[[BayesianActionSelector], str]) -> float:
        random.seed(seed)
        selector = BayesianActionSelector({k: (1.0, 1.0) for k in true_success_probs})
        for _ in range(steps):
            action = select_fn(selector)
            outcome = 1 if random.random() < true_success_probs[action] else 0
            selector.update(action, outcome)
        return sum(b.entropy() for b in selector.beliefs.values())

    return {
        "information_gain": run(lambda s: s.select_action()),
        "random": run(lambda s: random.choice(list(s.beliefs.keys()))),
    }

__all__ = [
    "BayesianActionSelector",
    "BetaDistribution",
    "simulate_strategies",
]
