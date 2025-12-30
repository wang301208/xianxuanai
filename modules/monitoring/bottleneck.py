from __future__ import annotations

"""Utilities for detecting bottlenecks using sliding window analysis."""

from collections import deque, defaultdict
from typing import Deque, Dict, Iterable, Tuple


class BottleneckDetector:
    """Identify hotspot modules based on recent latency samples."""

    def __init__(self, window_size: int = 50) -> None:
        self.window_size = window_size
        self._windows: Dict[str, Deque[float]] = defaultdict(deque)

    def record(self, module: str, latency: float) -> None:
        """Record ``latency`` for ``module``."""
        window = self._windows[module]
        window.append(latency)
        if len(window) > self.window_size:
            window.popleft()

    def _averages(self) -> Iterable[Tuple[str, float]]:
        for module, window in self._windows.items():
            if window:
                yield module, sum(window) / len(window)

    def bottleneck(self) -> Tuple[str, float] | None:
        """Return the module with highest average latency and its value."""
        averages = list(self._averages())
        if not averages:
            return None
        return max(averages, key=lambda x: x[1])

    def all_averages(self) -> Dict[str, float]:
        """Return average latency for each module."""
        return {module: avg for module, avg in self._averages()}
