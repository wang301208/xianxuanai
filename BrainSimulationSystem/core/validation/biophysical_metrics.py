"""Metrics helpers for the biophysical (downscaled whole-brain) backend.

These utilities are dependency-free (NumPy only) and intended for quick system-level
sanity checks and lightweight calibration loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np


@dataclass
class BiophysicalRunSummary:
    dt_ms: float
    duration_ms: float
    n_neurons: int
    total_spikes: int
    mean_rate_hz: float
    region_mean_rates_hz: Dict[str, float]
    dominant_population_oscillation_hz: Optional[float]


def spike_count_series(states: Iterable[Dict[str, Any]]) -> np.ndarray:
    counts = []
    for state in states:
        if not isinstance(state, dict):
            continue
        if "spike_count" in state:
            counts.append(int(state.get("spike_count") or 0))
        else:
            counts.append(int(len(state.get("spikes") or [])))
    return np.asarray(counts, dtype=np.int32)


def mean_firing_rate_hz(*, total_spikes: int, n_neurons: int, duration_ms: float) -> float:
    if n_neurons <= 0:
        return 0.0
    duration_s = float(duration_ms) / 1000.0
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        return 0.0
    return float(total_spikes) / (float(n_neurons) * duration_s)


def region_mean_rates_hz(
    states: Iterable[Dict[str, Any]], *, region_sizes: Dict[str, int], dt_ms: float
) -> Dict[str, float]:
    totals = {name: 0 for name in region_sizes}
    steps = 0
    for state in states:
        if not isinstance(state, dict):
            continue
        per_region = state.get("region_spike_counts")
        if not isinstance(per_region, dict):
            continue
        for region, count in per_region.items():
            if region in totals:
                totals[region] += int(count or 0)
        steps += 1

    duration_s = float(steps) * float(dt_ms) / 1000.0
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        return {name: 0.0 for name in region_sizes}

    out: Dict[str, float] = {}
    for region, total_spikes in totals.items():
        n = int(region_sizes.get(region) or 0)
        out[region] = mean_firing_rate_hz(total_spikes=total_spikes, n_neurons=n, duration_ms=duration_s * 1000.0)
    return out


def dominant_oscillation_hz(
    series: np.ndarray, *, dt_ms: float, min_hz: float = 1.0, max_hz: float = 200.0
) -> Optional[float]:
    x = np.asarray(series, dtype=np.float32).reshape(-1)
    if x.size < 8:
        return None
    dt_s = float(dt_ms) / 1000.0
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        return None

    x = x - float(np.mean(x))
    spec = np.fft.rfft(x)
    power = (spec.real**2 + spec.imag**2).astype(np.float64)
    freqs = np.fft.rfftfreq(x.size, d=dt_s).astype(np.float64)

    mask = (freqs >= float(min_hz)) & (freqs <= float(max_hz))
    if not np.any(mask):
        return None

    idx = int(np.argmax(power[mask]))
    return float(freqs[mask][idx])


def summarize_run(
    states: Iterable[Dict[str, Any]], *, dt_ms: float, n_neurons: int, region_sizes: Dict[str, int]
) -> BiophysicalRunSummary:
    states_list = list(states)
    series = spike_count_series(states_list)
    total = int(np.sum(series)) if series.size else 0
    duration_ms = float(series.size) * float(dt_ms)
    mean_rate = mean_firing_rate_hz(total_spikes=total, n_neurons=n_neurons, duration_ms=duration_ms)
    region_rates = region_mean_rates_hz(states_list, region_sizes=region_sizes, dt_ms=dt_ms)
    osc = dominant_oscillation_hz(series, dt_ms=dt_ms)
    return BiophysicalRunSummary(
        dt_ms=float(dt_ms),
        duration_ms=float(duration_ms),
        n_neurons=int(n_neurons),
        total_spikes=int(total),
        mean_rate_hz=float(mean_rate),
        region_mean_rates_hz=region_rates,
        dominant_population_oscillation_hz=osc,
    )

