from __future__ import annotations

import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from evolution.ga_config import GAConfig  # noqa: E402
from evolution.parallel_ga import ParallelGA  # noqa: E402


def fitness(individual):
    """Artificially expensive fitness function for benchmarking."""
    time.sleep(0.05)
    return -sum((x - 0.5) ** 2 for x in individual)


def run_ga(parallel: bool) -> float:
    workers = 4 if parallel else 1
    config = GAConfig(population_size=20, max_generations=20, n_workers=workers)
    ga = ParallelGA(fitness, config)
    start = time.time()
    ga.run()
    return time.time() - start


def main() -> None:
    seq_time = run_ga(parallel=False)
    par_time = run_ga(parallel=True)
    speedup = seq_time / par_time if par_time else float("inf")
    print(f"Sequential time: {seq_time:.2f}s")
    print(f"Parallel time: {par_time:.2f}s")
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
