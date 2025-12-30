import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.neuromorphic.temporal_encoding import (
    latency_encode,
    rate_encode,
    decode_spike_counts,
    decode_average_rate,
)


def test_rate_encode_steps():
    encoded = rate_encode([1.0, 0.0], steps=3)
    assert len(encoded) == 3
    assert encoded[0][0] == 1
    assert encoded[1][0] == 1
    assert encoded[2][0] == 1
    assert all(spike == 0 for step in encoded for spike in step[1:])


def test_latency_encode_order():
    events = latency_encode([0.2, 0.9], t_scale=1.0)
    times = [time for time, _ in events]
    assert times == sorted(times)


def test_decode_helpers():
    outputs = [(0.0, [1, 0]), (1.0, [0, 1]), (2.0, [1, 1])]
    counts = decode_spike_counts(outputs)
    assert counts == [2, 2]
    rates = decode_average_rate(outputs, window=2.0)
    assert rates == [1.0, 1.0]
