"""Trend analytics for founder agent."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict

from .intrinsic_reward import compute_intrinsic_reward


class Analytics:
    """Maintain rolling metric history and compute simple trends."""

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self._history: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=window)
        )

    async def handle_event(self, event: Dict[str, float]) -> None:
        """Process incoming metric event."""
        for key, value in event.items():
            if isinstance(value, (int, float)):
                self._history[key].append(float(value))

    def get_trends(self) -> Dict[str, float]:
        """Return the difference between newest and oldest values per metric."""
        trends: Dict[str, float] = {}
        for key, values in self._history.items():
            if len(values) >= 2:
                trends[key] = values[-1] - values[0]
            elif values:
                trends[key] = 0.0
        return trends

    def get_intrinsic_reward(self) -> float:
        """Calculate an intrinsic reward based on current metric trends."""
        return compute_intrinsic_reward(self.get_trends())
