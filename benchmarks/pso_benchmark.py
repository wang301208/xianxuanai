"""Compare static vs. adaptive Particle Swarm Optimization."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pso import pso, linear_schedule


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def run(static: bool) -> float:
    bounds = [(-5, 5)] * 3
    if static:
        result = pso(
            sphere,
            bounds,
            max_iter=100,
            w_schedule=lambda _, w: w,
        )
    else:
        sched_w = linear_schedule(0.9, 0.4, 100)
        sched_c1 = linear_schedule(2.5, 0.5, 100)
        sched_c2 = linear_schedule(0.5, 2.5, 100)
        result = pso(
            sphere,
            bounds,
            max_iter=100,
            w_schedule=sched_w,
            c1_schedule=sched_c1,
            c2_schedule=sched_c2,
            log_params=True,
        )
    return result


def main() -> None:
    static_res = run(static=True)
    adaptive_res = run(static=False)
    print(f"Static PSO best value:    {static_res.value:.4g}")
    print(f"Adaptive PSO best value:  {adaptive_res.value:.4g}")
    print(f"Logged {len(adaptive_res.w_history)} iterations of parameters")


if __name__ == "__main__":
    main()
