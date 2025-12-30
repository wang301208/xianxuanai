"""Configuration objects for A/B algorithm tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence, Tuple


@dataclass
class ABTestConfig:
    """Define an A/B experiment setup.

    Attributes:
        algo_a: Callable implementing algorithm variant A.
        algo_b: Callable implementing algorithm variant B.
        data:   Tuple of features and ground-truth labels.
        name_a: Optional display name for algorithm A.
        name_b: Optional display name for algorithm B.
    """

    algo_a: Callable[[Any, Sequence[int]], Sequence[int]]
    algo_b: Callable[[Any, Sequence[int]], Sequence[int]]
    data: Tuple[Any, Sequence[int]]
    name_a: str = "algo_a"
    name_b: str = "algo_b"
