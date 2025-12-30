from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def latency_encode(
    signal: List[float],
    *,
    t_start: float = 0.0,
    t_scale: float = 1.0,
) -> List[Tuple[float, List[int]]]:
    """Encode an analog input vector into spike-time events using latency coding.

    Each value in ``signal`` is interpreted as an amplitude in the range ``[0, 1]``.
    Larger values produce earlier spikes according to
    ``time = t_start + t_scale * (1 - value)``.

    Parameters
    ----------
    signal
        Analog input vector. Values are clamped to ``[0, 1]``.
    t_start
        Base timestamp for all spikes.
    t_scale
        Scaling factor mapping amplitudes to spike latencies.

    Returns
    -------
    list of tuple
        A list of ``(time, spikes)`` pairs sorted by time. ``spikes`` is a list
        with the same length as ``signal`` containing ``1`` at the index of the
        neuron that spikes and ``0`` elsewhere.
    """

    n = len(signal)
    events: Dict[float, List[int]] = {}
    for i, value in enumerate(signal):
        clamped = max(0.0, min(1.0, value))
        time = t_start + t_scale * (1.0 - clamped)
        spikes = events.setdefault(time, [0] * n)
        spikes[i] = 1

    return sorted(events.items())


def rate_encode(signal: Sequence[float], *, steps: int = 5) -> List[List[int]]:
    """Encode analog values as rate-coded spike trains.

    Parameters
    ----------
    signal:
        Sequence of analog values in [0, 1]. Values outside the range are
        clamped.
    steps:
        Number of discrete timesteps used to represent the rate.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    trains: List[List[int]] = []
    for _ in range(steps):
        trains.append([0] * len(signal))
    for neuron, value in enumerate(signal):
        clamped = max(0.0, min(1.0, value))
        spikes = int(round(clamped * steps))
        for step in range(spikes):
            trains[step][neuron] = 1
    return trains


def decode_spike_counts(outputs: Sequence[Tuple[float, List[int]]]) -> List[int]:
    """Aggregate spike counts per neuron from network outputs."""

    if not outputs:
        return []
    n = len(outputs[0][1])
    counts = [0] * n
    for _, spikes in outputs:
        for idx, spike in enumerate(spikes):
            counts[idx] += spike
    return counts


def decode_average_rate(outputs: Sequence[Tuple[float, List[int]]], *, window: float = 1.0) -> List[float]:
    """Compute average firing rate per neuron assuming a fixed window."""

    counts = decode_spike_counts(outputs)
    if not counts:
        return []
    if window <= 0:
        raise ValueError("window must be positive")
    return [count / window for count in counts]
