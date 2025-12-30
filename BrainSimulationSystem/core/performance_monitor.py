"""
Performance monitoring and adaptive scheduling utilities.

Provides a small analytics layer that aggregates recent runtime
statistics and emits recommendations for coarse-grained scheduling
adjustments (e.g. toggling region fidelity or enabling GPU acceleration).
"""

from __future__ import annotations

import statistics
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Optional, Tuple


@dataclass
class PerformancePolicy:
    window: int = 50
    max_region_time: float = 0.03  # seconds
    min_region_time: float = 0.005
    max_global_time: float = 0.1
    evaluation_interval: int = 10


class PerformanceMonitor:
    """Aggregates timing metrics and synthesises adaptive scheduling actions."""

    def __init__(self, policy: Optional[Dict[str, float]] = None):
        self.policy = PerformancePolicy(**(policy or {}))
        self._global_history: Deque[float] = deque(maxlen=self.policy.window)
        self._region_history: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.policy.window)
        )
        self._step_counter = 0
        self._last_actions: Dict[str, Dict[str, str]] = {}

    # ------------------------------------------------------------------ logging
    def record_global_step(self, elapsed: float) -> None:
        self._global_history.append(float(elapsed))
        self._step_counter += 1

    def record_region_step(self, region: str, elapsed: float) -> None:
        self._region_history[region].append(float(elapsed))

    # ---------------------------------------------------------------- analysis
    def should_evaluate(self) -> bool:
        return self._step_counter % max(1, self.policy.evaluation_interval) == 0

    def generate_actions(self) -> Dict[str, Dict[str, str]]:
        """Return scheduling recommendations based on recent history."""

        if not self._global_history:
            return {}

        actions: Dict[str, Dict[str, str]] = {"region_modes": {}, "gpu": {}}

        global_mean = _safe_mean(self._global_history)
        if global_mean and global_mean > self.policy.max_global_time:
            actions["gpu"]["enable"] = "true"

        for region, history in self._region_history.items():
            if len(history) < max(3, self.policy.window // 3):
                continue
            avg_time = _safe_mean(history)
            if avg_time is None:
                continue
            if avg_time > self.policy.max_region_time:
                actions["region_modes"][region] = "macro"
            elif avg_time < self.policy.min_region_time:
                actions["region_modes"][region] = "micro"

        # Remove empty groups
        actions = {k: v for k, v in actions.items() if v}
        self._last_actions = actions
        return actions

    def last_actions(self) -> Dict[str, Dict[str, str]]:
        return self._last_actions


def _safe_mean(values: Iterable[float]) -> Optional[float]:
    sample = list(values)
    if not sample:
        return None
    try:
        return statistics.fmean(sample)
    except Exception:  # pragma: no cover - statistics fallback
        return sum(sample) / len(sample)


__all__ = ["PerformanceMonitor", "PerformancePolicy"]

