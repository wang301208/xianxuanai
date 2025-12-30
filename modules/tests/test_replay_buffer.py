import sys
import os
from collections import Counter

import pytest

sys.path.insert(0, os.path.abspath(os.getcwd()))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "modules")))

from evolution.replay_buffer import ReplayBuffer


def test_uniform_sampling_distribution() -> None:
    buf = ReplayBuffer(capacity=10)
    for i in range(10):
        buf.push(i)
    counts = Counter()
    for _ in range(1000):
        (sample,), _, _ = buf.sample(1)
        counts[sample[0]] += 1  # type: ignore[index]
    # roughly uniform distribution: max/min ratio should not be extreme
    assert max(counts.values()) - min(counts.values()) < 300


def test_prioritized_sampling_bias() -> None:
    buf = ReplayBuffer(capacity=5, prioritized=True)
    for i in range(5):
        buf.push(i)
    # give last element much higher priority
    buf.update_priorities(list(range(5)), [1, 1, 1, 1, 100])
    counts = Counter()
    for _ in range(500):
        (sample,), _, _ = buf.sample(1)
        counts[sample[0]] += 1  # type: ignore[index]
    assert counts[4] > max(counts[i] for i in range(4))
