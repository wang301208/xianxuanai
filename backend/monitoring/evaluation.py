"""Multi-metric evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EvaluationMetrics:
    """Tracks classification and performance metrics."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    latencies: List[float] = field(default_factory=list)
    group_positive: Dict[str, int] = field(default_factory=dict)
    group_total: Dict[str, int] = field(default_factory=dict)

    def record(self, prediction: bool, truth: bool, latency: float, group: str) -> None:
        """Record a single prediction outcome."""
        if prediction and truth:
            self.tp += 1
        elif prediction and not truth:
            self.fp += 1
        elif not prediction and truth:
            self.fn += 1
        else:
            self.tn += 1

        self.latencies.append(latency)
        self.group_total[group] = self.group_total.get(group, 0) + 1
        if prediction:
            self.group_positive[group] = self.group_positive.get(group, 0) + 1

    # metric helpers -----------------------------------------------------
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    def latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    def fairness(self) -> float:
        """Return absolute difference in positive rates across groups."""
        rates: List[float] = []
        for g, total in self.group_total.items():
            pos = self.group_positive.get(g, 0)
            rates.append(pos / total if total else 0.0)
        if not rates:
            return 0.0
        return max(rates) - min(rates)

    def summary(self) -> Dict[str, float]:
        return {
            "precision": self.precision(),
            "recall": self.recall(),
            "latency": self.latency(),
            "fairness": self.fairness(),
        }
