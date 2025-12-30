from benchmarks.performance_monitor_benchmark import run_benchmark


def test_performance_monitor_benchmark() -> None:
    metrics = run_benchmark(10_000)
    assert metrics["throughput_after"] >= metrics["throughput_before"]

