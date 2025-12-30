"""Tests for RollingAnomalyModel integrated with PerformanceDiagnoser."""

from modules.monitoring import RollingAnomalyModel, PerformanceDiagnoser, MetricEvent


def test_anomaly_model_flags_high_latency():
    model = RollingAnomalyModel()
    diagnoser = PerformanceDiagnoser(anomaly_model=model, anomaly_threshold=1.0, max_latency_s=10.0)
    events = [
        MetricEvent(module="a", latency=0.1, energy=0.0, throughput=1.0, timestamp=0.0),
        MetricEvent(module="a", latency=5.0, energy=0.0, throughput=1.0, timestamp=1.0),
    ]
    report = diagnoser.diagnose(events)
    kinds = {issue.kind for issue in report["issues"]}
    assert "anomaly" in kinds
