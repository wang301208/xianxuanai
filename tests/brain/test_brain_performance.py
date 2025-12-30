import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modules.brain.performance import (
    EnergyConsumptionProfiler,
    LatencyProfiler,
    auto_optimize_performance,
)


def test_energy_consumption_profiler() -> None:
    profiler = EnergyConsumptionProfiler(voltage=2.0)
    currents = [1.0, 2.0, 3.0]
    result = profiler.profile(currents)
    assert result["total_energy"] == pytest.approx(12.0)
    assert result["average_power"] == pytest.approx(4.0)


def test_latency_profiler() -> None:
    profiler = LatencyProfiler()
    latencies = [0.1, 0.2, 0.3]
    result = profiler.profile(latencies)
    assert result["average_latency"] == pytest.approx(0.2)
    assert result["max_latency"] == pytest.approx(0.3)


def test_auto_optimize_performance() -> None:
    spikes = [0.1, 0.2, 0.3]
    currents = [5.0, 5.0, 5.0]  # High energy usage
    latencies = [1.0, 1.2]  # High latency
    outcome = auto_optimize_performance(spikes, currents, latencies)
    assert any("energy" in s for s in outcome["suggestions"])
    assert any("latency" in s.lower() for s in outcome["suggestions"])
    assert len(outcome["adjustments"]) == 2
