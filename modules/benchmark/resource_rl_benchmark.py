"""Benchmark CPU vs GPU training and evaluation for ResourceRL."""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import torch

from evolution.resource_rl import ResourceRL, Transition


def _generate_transitions(n: int = 1000) -> list[Transition]:
    """Create synthetic transitions for benchmarking."""

    rng = np.random.default_rng(0)
    states = rng.uniform(0, 100, size=(n, 2)).astype(np.float32)
    agent = ResourceRL()
    transitions: list[Transition] = []
    for state in states:
        for action in range(3):
            reward = agent._rule_reward(state, action)
            transitions.append(Transition(state, action, reward))
    return transitions


def _benchmark(device: str) -> None:
    agent = ResourceRL(device=device)
    transitions = _generate_transitions()

    start = time.perf_counter()
    agent.train(transitions, epochs=10)
    train_time = time.perf_counter() - start

    sample_states: Iterable[tuple[float, float]] = (
        (float(i % 100), float((i * 2) % 100)) for i in range(1000)
    )
    start = time.perf_counter()
    agent.evaluate(sample_states)
    eval_time = time.perf_counter() - start

    print(f"{device} train time: {train_time:.3f}s, eval time: {eval_time:.3f}s")


if __name__ == "__main__":
    _benchmark("cpu")
    if torch.cuda.is_available():
        _benchmark("cuda")
    else:
        print("CUDA/ROCm GPU not available; skipping GPU benchmark.")

