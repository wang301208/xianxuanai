"""Execution utilities for running A/B algorithm comparisons."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List

from .config import ABTestConfig


@dataclass
class AlgorithmResult:
    """Metrics collected for a single algorithm run."""

    name: str
    accuracy: float
    duration: float
    correctness: List[int]


@dataclass
class ABTestResult:
    """Results for both algorithm variants."""

    algo_a: AlgorithmResult
    algo_b: AlgorithmResult


def _evaluate(algo, X, y):
    start = time.perf_counter()
    preds = algo(X, y)
    duration = time.perf_counter() - start
    correctness = [int(p == t) for p, t in zip(preds, y)]
    accuracy = sum(correctness) / len(correctness)
    return AlgorithmResult(algo.__name__, accuracy, duration, correctness)


def run_ab_test(config: ABTestConfig) -> ABTestResult:
    """Run both algorithms in parallel and collect metrics."""

    X, y = config.data
    with ThreadPoolExecutor(max_workers=2) as ex:
        future_a = ex.submit(_evaluate, config.algo_a, X, y)
        future_b = ex.submit(_evaluate, config.algo_b, X, y)
        res_a = future_a.result()
        res_b = future_b.result()

    # Override names if provided in config
    res_a.name = config.name_a
    res_b.name = config.name_b
    return ABTestResult(res_a, res_b)
