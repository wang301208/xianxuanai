"""Adaptive fitness generator for multi-objective optimization.

This module provides :class:`AdaptiveFitnessGenerator`, which automatically
learns the weights of multiple fitness metrics from historical performance and
external environment signals.  The generator can be used as a drop-in fitness
function for the :class:`~modules.evolution.generic_ga.GeneticAlgorithm`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence


@dataclass
class AdaptiveFitnessGenerator:
    """Combine multiple metrics into a single adaptive fitness score.

    Parameters
    ----------
    metrics:
        Sequence of ``(name, callable)`` metric functions.
    learning_rate:
        Controls how quickly weights adapt to new information.
    decay:
        Exponential decay factor for the running average of metric scores.  A
        value close to ``1.0`` gives long memory while smaller values react more
        quickly to recent performance.
    """

    metrics: Sequence[tuple[str, Callable[[Sequence[float]], float]]]
    learning_rate: float = 0.1
    decay: float = 0.9
    history: list[Dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.metrics:
            raise ValueError("metrics must not be empty")
        # Start with equal weights for each metric.
        initial = 1.0 / len(self.metrics)
        self.weights: Dict[str, float] = {name: initial for name, _ in self.metrics}
        # Exponential moving average of scores for each metric.
        self.avg_scores: Dict[str, float] = {name: 0.0 for name, _ in self.metrics}

    # ------------------------------------------------------------------
    def __call__(
        self, individual: Sequence[float], env_priorities: Dict[str, float] | None = None
    ) -> float:
        """Evaluate ``individual`` and update internal weights.

        ``env_priorities`` is a mapping of metric names to relative importance
        provided by the environment.  Missing metrics default to ``1.0``.  The
        method returns the combined fitness score.
        """

        scores: Dict[str, float] = {}
        for name, fn in self.metrics:
            score = fn(individual)
            scores[name] = score
            # Update running average for historical performance.
            self.avg_scores[name] = self.decay * self.avg_scores[name] + (1 - self.decay) * score

        fitness = sum(self.weights[name] * scores[name] for name in scores)
        self.history.append({**{f"w_{k}": v for k, v in self.weights.items()}, **scores, "fitness": fitness})

        if env_priorities is not None:
            self._update(env_priorities)

        return fitness

    # ------------------------------------------------------------------
    def _update(self, env_priorities: Dict[str, float]) -> None:
        """Update metric weights using environment priorities and past scores."""

        for name in self.weights:
            priority = env_priorities.get(name, 1.0)
            avg_score = self.avg_scores.get(name, 0.0)
            # Emphasize metrics that are important (high priority) yet underperforming
            adjustment = priority / (abs(avg_score) + 1e-8)
            self.weights[name] = (1 - self.learning_rate) * self.weights[name] + self.learning_rate * adjustment

        # Normalize weights so they sum to one.
        total = sum(self.weights.values())
        if total > 0:
            for name in self.weights:
                self.weights[name] /= total
