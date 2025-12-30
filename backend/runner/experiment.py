"""Utility to run algorithms on benchmark problems with multiple seeds."""
from __future__ import annotations
from typing import Iterable, Dict, Any, List

import yaml

from algorithms import ALGORITHMS
from benchmarks import PROBLEMS
from metrics.recorder import MetricsRecorder
from modules.diagnostics import record_error
from .logging_config import get_logger


logger = get_logger(__name__)


def _load_config(path: str = "config/experiment.yaml") -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def run_experiments(
    algorithm: str,
    problem: str,
    seeds: Iterable[int],
    max_iters: int | None = None,
    max_time: float | None = None,
    patience: int | None = None,
    output: str | None = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Run ``algorithm`` on ``problem`` for each seed.

    If ``output`` is provided, metrics are written to the given JSON or CSV file.
    """
    config = _load_config()
    max_iters = max_iters or config.get("max_iters", 100)
    max_time = max_time or config.get("time_budget")
    patience = patience or config.get("patience")

    algo_fn = ALGORITHMS[algorithm]
    problem_cls = PROBLEMS[problem]
    prob = problem_cls()

    recorder = MetricsRecorder()
    for seed in seeds:
        try:
            best_x, best_val, iterations, elapsed = algo_fn(
                prob,
                seed=seed,
                max_iters=max_iters,
                max_time=max_time,
                patience=patience,
                **kwargs,
            )
        except Exception as e:  # pragma: no cover - exceptional path
            record_error(
                e,
                {
                    "algorithm": algorithm,
                    "problem": prob.name,
                    "seed": seed,
                },
            )
            raise
        recorder.record(
            algorithm,
            prob.name,
            seed,
            best_val,
            prob.optimum_value,
            iterations,
            elapsed,
            max_iters=max_iters,
            max_time=max_time,
        )

    if output:
        recorder.save(output)

    return recorder.to_list()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run optimization experiments")
    parser.add_argument("algorithm", choices=ALGORITHMS.keys())
    parser.add_argument("problem", choices=PROBLEMS.keys())
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--output", type=str, default=None, help="JSON or CSV output path")
    args = parser.parse_args()

    output = run_experiments(
        args.algorithm,
        args.problem,
        args.seeds,
        max_iters=args.max_iters,
        max_time=args.max_time,
        patience=args.patience,
        output=args.output,
    )
    if not args.output:
        logger.info("experiment_output", output=output)
