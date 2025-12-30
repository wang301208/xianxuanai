"""Command line interface for running A/B algorithm comparisons."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Callable

from sklearn.datasets import make_classification

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.ab_testing import (
    ABTestConfig,
    confidence_interval,
    run_ab_test,
    significance_test,
)


def _load_callable(path: str) -> Callable:
    module_name, func_name = path.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A/B algorithm test")
    parser.add_argument("--algoA", required=True, help="module:function for algorithm A")
    parser.add_argument("--algoB", required=True, help="module:function for algorithm B")
    parser.add_argument("--size", type=int, default=200, help="dataset size")
    args = parser.parse_args()

    X, y = make_classification(
        n_samples=args.size, n_features=20, n_informative=15, random_state=0
    )

    algo_a = _load_callable(args.algoA)
    algo_b = _load_callable(args.algoB)
    config = ABTestConfig(algo_a=algo_a, algo_b=algo_b, data=(X, y), name_a=args.algoA, name_b=args.algoB)

    result = run_ab_test(config)
    diff = [a - b for a, b in zip(result.algo_a.correctness, result.algo_b.correctness)]
    ci_low, ci_high = confidence_interval(diff)
    t_stat, p_val = significance_test(result.algo_a.correctness, result.algo_b.correctness)

    print(f"{result.algo_a.name}: accuracy={result.algo_a.accuracy:.3f} time={result.algo_a.duration:.3f}s")
    print(f"{result.algo_b.name}: accuracy={result.algo_b.accuracy:.3f} time={result.algo_b.duration:.3f}s")
    print(f"Accuracy diff (A-B): {result.algo_a.accuracy - result.algo_b.accuracy:.3f}")
    print(f"95% CI for diff: ({ci_low:.3f}, {ci_high:.3f})")
    print(f"t-statistic={t_stat:.3f}, p-value={p_val:.3f}")


if __name__ == "__main__":
    main()
