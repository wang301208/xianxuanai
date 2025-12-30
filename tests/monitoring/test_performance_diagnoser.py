"""Tests for PerformanceDiagnoser."""

from modules.monitoring import PerformanceDiagnoser, MetricEvent


def _event(module, latency=1.0, throughput=1.0, energy=0.5, status="success"):
    return MetricEvent(
        module=module,
        latency=latency,
        throughput=throughput,
        energy=energy,
        timestamp=0.0,
        status=status,
    )


def test_diagnoser_detects_latency_and_success():
    diagnoser = PerformanceDiagnoser(max_latency_s=0.5, min_success_rate=0.8)
    events = [
        _event("a", latency=0.6, status="success"),
        _event("a", latency=0.7, status="failure"),
        _event("b", latency=0.2, status="success"),
    ]

    report = diagnoser.diagnose(events)

    kinds = {issue.kind for issue in report["issues"]}
    assert "high_latency" in kinds
    assert "low_success_rate" in kinds
    assert report["modules"]["a"]["success_rate"] < 0.8
