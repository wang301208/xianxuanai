from __future__ import annotations

"""Cognitive benchmark utilities for evolutionary optimisation."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_WEIGHTS: Dict[str, float] = {
    "success": 0.6,
    "efficiency": 0.2,
    "quality": 0.15,
    "reward": 0.05,
}


@dataclass
class CognitiveBenchmarkResult:
    """Outcome produced by evaluating the agent on a benchmark task."""

    task_id: str
    success: bool
    latency: float
    reward: float = 0.0
    quality: float = 0.0
    steps: int = 0
    tokens_used: int = 0
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Compute a weighted scalar score for this benchmark result."""

        weights = (weights or DEFAULT_WEIGHTS).copy()
        _normalise_weights(weights)

        success_score = 1.0 if self.success else 0.0
        efficiency_score = 0.0
        if self.latency > 0:
            efficiency_score = min(1.0, 1.0 / (self.latency + 1e-6))
        quality_score = max(0.0, min(1.0, self.quality))
        reward_score = max(-1.0, min(1.0, self.reward))

        return (
            weights["success"] * success_score
            + weights["efficiency"] * efficiency_score
            + weights["quality"] * quality_score
            + weights["reward"] * reward_score
        )


def _normalise_weights(weights: Dict[str, float]) -> None:
    total = sum(max(v, 0.0) for v in weights.values())
    if total <= 0:
        weights.update(DEFAULT_WEIGHTS)
        total = sum(weights.values())
    for key in list(weights.keys()):
        weights[key] = max(weights[key], 0.0) / total


def summarise_benchmarks(
    results: Iterable[CognitiveBenchmarkResult],
) -> Dict[str, float]:
    """Produce aggregate statistics for the supplied benchmark results."""

    results_list: List[CognitiveBenchmarkResult] = list(results)
    if not results_list:
        return {"success_rate": 0.0, "avg_latency": 0.0, "avg_quality": 0.0, "avg_reward": 0.0}

    total = len(results_list)
    success_rate = sum(1 for r in results_list if r.success) / total
    avg_latency = sum(r.latency for r in results_list) / total
    avg_quality = sum(r.quality for r in results_list) / total
    avg_reward = sum(r.reward for r in results_list) / total

    return {
        "success_rate": success_rate,
        "avg_latency": avg_latency,
        "avg_quality": avg_quality,
        "avg_reward": avg_reward,
        "task_count": float(total),
    }


def aggregate_benchmark_score(
    results: Iterable[CognitiveBenchmarkResult],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Aggregate benchmark results into a single scalar fitness value."""

    weights = (weights or DEFAULT_WEIGHTS).copy()
    _normalise_weights(weights)
    results_list = list(results)
    if not results_list:
        return 0.0

    scores = [sample.score(weights) for sample in results_list]
    return sum(scores) / len(scores)

