"""Lightweight anomaly scorer for metric events."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable
import math

from .collector import MetricEvent


class RollingAnomalyModel:
    """Z-score style anomaly scoring per module."""

    def __init__(self, window: int = 50) -> None:
        self.window = window
        self._sums: Dict[str, float] = defaultdict(float)
        self._sumsq: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)

    def score_events(self, events: Iterable[MetricEvent]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for event in events:
            module = event.module
            latency = float(getattr(event, "latency", 0.0))
            count = self._counts[module]
            mean = self._sums[module] / count if count else 0.0
            var = (self._sumsq[module] / count - mean * mean) if count else 0.0
            std = math.sqrt(max(var, 1e-9))
            z = abs(latency - mean) / std if std > 0 else 0.0
            scores[module] = max(scores.get(module, 0.0), z)
            # Update rolling stats
            if count >= self.window:
                # decay oldest via simple shrink
                self._sums[module] *= 0.9
                self._sumsq[module] *= 0.9
                self._counts[module] = int(self._counts[module] * 0.9)
            self._sums[module] += latency
            self._sumsq[module] += latency * latency
            self._counts[module] += 1
        return scores


__all__ = ["RollingAnomalyModel"]
