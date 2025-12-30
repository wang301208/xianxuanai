"""Tests for the cross-domain benchmark suite."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from benchmarks.run_cross_domain import CrossDomainBenchmark


def test_cross_domain_benchmark_adapts_strategy() -> None:
    benchmark = CrossDomainBenchmark()
    results = benchmark.run()

    assert len(results) == 2
    assert {r.domain for r in results} == {"logic", "knowledge"}
    # Ensure that the strategy changed after the first task
    assert results[0].strategy != results[1].strategy
