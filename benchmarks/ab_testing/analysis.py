"""Statistical tools for comparing A/B test results."""

from __future__ import annotations

import math
import statistics
from typing import Iterable, Tuple


def confidence_interval(samples: Iterable[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Return the normal confidence interval for the given samples."""

    data = list(samples)
    dist = statistics.NormalDist.from_samples(data)
    z = statistics.NormalDist().inv_cdf(0.5 + confidence / 2)
    margin = z * dist.stdev / math.sqrt(len(data))
    return dist.mean - margin, dist.mean + margin


def significance_test(a: Iterable[int], b: Iterable[int]) -> Tuple[float, float]:
    """Paired t-test for per-sample metric differences.

    Returns the t statistic and two-tailed p-value.
    """

    differences = [x - y for x, y in zip(a, b)]
    if len(differences) < 2:
        return float("nan"), float("nan")

    mean_diff = statistics.mean(differences)
    sd_diff = statistics.stdev(differences)
    t_stat = mean_diff / (sd_diff / math.sqrt(len(differences)))
    p_value = 2 * statistics.NormalDist().cdf(-abs(t_stat))
    return t_stat, p_value
