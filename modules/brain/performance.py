"""Performance monitoring utilities for brain modules.

This module defines light-weight components for analyzing spike
patterns, energy consumption and latency within neural simulations.
It also exposes helper functions to profile performance and
apply simplistic optimisation strategies based on the gathered
metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class SpikePatternAnalyzer:
    """Analyse basic statistics of neural spike trains."""

    threshold: float = 0.0

    def analyse(self, spikes: List[float]) -> Dict[str, float]:
        """Return statistics about *spikes*.

        Parameters
        ----------
        spikes:
            List of spike magnitudes.
        """

        if not spikes:
            return {"mean": 0.0, "max": 0.0, "above_threshold": 0.0}
        mean = sum(spikes) / len(spikes)
        maximum = max(spikes)
        above = sum(1 for s in spikes if s > self.threshold) / len(spikes)
        return {"mean": mean, "max": maximum, "above_threshold": above}


@dataclass
class EnergyConsumptionProfiler:
    """Estimate energy usage from current measurements."""

    voltage: float = 1.0

    def profile(self, current_samples: List[float]) -> Dict[str, float]:
        """Profile energy from *current_samples*.

        Energy is computed as ``voltage * sum(current_samples)`` which
        assumes a unit time step between samples.  The mean power is
        ``voltage * mean(current_samples)``.
        """

        if not current_samples:
            return {"total_energy": 0.0, "average_power": 0.0}
        total_current = sum(current_samples)
        mean_current = total_current / len(current_samples)
        total_energy = self.voltage * total_current
        average_power = self.voltage * mean_current
        return {
            "total_energy": total_energy,
            "average_power": average_power,
        }


@dataclass
class LatencyProfiler:
    """Compute latency statistics for processing pipelines."""

    def profile(self, latencies: List[float]) -> Dict[str, float]:
        """Return mean and max latency for a sequence of measurements."""

        if not latencies:
            return {"average_latency": 0.0, "max_latency": 0.0}
        avg = sum(latencies) / len(latencies)
        mx = max(latencies)
        return {"average_latency": avg, "max_latency": mx}


def profile_brain_performance(
    spikes: List[float],
    currents: List[float],
    latencies: List[float],
    *,
    spike_threshold: float = 0.0,
    voltage: float = 1.0,
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Profile various performance metrics of a brain simulation.

    Returns a tuple ``(metrics, suggestions)`` where ``metrics`` contains
    dictionaries for each analysed aspect and ``suggestions`` lists
    simple textual recommendations based on those metrics.
    """

    spike_analyser = SpikePatternAnalyzer(threshold=spike_threshold)
    energy_profiler = EnergyConsumptionProfiler(voltage=voltage)
    latency_profiler = LatencyProfiler()

    metrics = {
        "spikes": spike_analyser.analyse(spikes),
        "energy": energy_profiler.profile(currents),
        "latency": latency_profiler.profile(latencies),
    }

    suggestions: List[str] = []
    if metrics["energy"]["total_energy"] > 10.0:
        suggestions.append("High energy consumption detected; reduce firing rates")
    if metrics["latency"]["average_latency"] > 0.5:
        suggestions.append("Latency is high; streamline processing pathways")
    return metrics, suggestions


def auto_optimize_performance(
    spikes: List[float],
    currents: List[float],
    latencies: List[float],
) -> Dict[str, object]:
    """Profile performance and apply naive optimisation steps.

    The function calls :func:`profile_brain_performance` and then returns
    a dictionary containing the metrics, suggestions and a record of the
    adjustments that would be executed.  The adjustments are symbolic and
    do not alter external state but demonstrate how the system might
    respond to the gathered metrics.
    """

    metrics, suggestions = profile_brain_performance(spikes, currents, latencies)

    adjustments: List[str] = []
    for suggestion in suggestions:
        if "energy" in suggestion:
            adjustments.append("Reducing neural activity to save energy")
        elif "Latency" in suggestion or "latency" in suggestion:
            adjustments.append("Reconfiguring pathways for lower latency")
    return {
        "metrics": metrics,
        "suggestions": suggestions,
        "adjustments": adjustments,
    }


__all__ = [
    "SpikePatternAnalyzer",
    "EnergyConsumptionProfiler",
    "LatencyProfiler",
    "profile_brain_performance",
    "auto_optimize_performance",
]
