"""Intrinsic reward calculation utilities for founder agent."""
from __future__ import annotations

from typing import Dict


def compute_intrinsic_reward(trends: Dict[str, float]) -> float:
    """Compute an intrinsic reward signal from metric trends.

    The default implementation encourages exploration by rewarding
    large changes in the agent's observed metrics, regardless of
    direction. This is a simple heuristic that can be replaced with
    a more sophisticated novelty or curiosity model in the future.

    Parameters
    ----------
    trends:
        Mapping of metric names to their recent trend values.

    Returns
    -------
    float
        The computed intrinsic reward.
    """
    return float(sum(abs(v) for v in trends.values()))
