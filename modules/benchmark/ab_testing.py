"""Simple A/B testing framework for benchmarking models or strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List


@dataclass
class ABTestResult:
    name_a: str
    name_b: str
    scores_a: List[float]
    scores_b: List[float]

    def summary(self) -> Dict[str, float]:
        """Return mean scores for both variants."""
        return {
            self.name_a: sum(self.scores_a) / len(self.scores_a) if self.scores_a else 0.0,
            self.name_b: sum(self.scores_b) / len(self.scores_b) if self.scores_b else 0.0,
        }


class ABTester:
    """Run the same tasks on two different callables and compare scores."""

    def __init__(self, scorer: Callable[[object], float]):
        """Create an ABTester.

        Args:
            scorer: Function that takes a result object and returns a numeric
                score representing performance on the task.
        """

        self.scorer = scorer

    def run(self, name_a: str, model_a: Callable[[object], object], name_b: str, model_b: Callable[[object], object], tasks: Iterable[object]) -> ABTestResult:
        """Execute tasks using both models and collect scores."""

        scores_a: List[float] = []
        scores_b: List[float] = []

        for task in tasks:
            result_a = model_a(task)
            result_b = model_b(task)
            scores_a.append(self.scorer(result_a))
            scores_b.append(self.scorer(result_b))

        return ABTestResult(name_a, name_b, scores_a, scores_b)
