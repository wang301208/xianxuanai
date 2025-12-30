"""Benchmark to measure throughput before and after a simple optimization."""

from __future__ import annotations

import time

import psutil

from backend.monitoring import (
    PerformanceMonitor,
    TimeSeriesStorage,
    dashboard_alert,
    email_alert,
)


def _baseline_workload(n: int) -> int:
    total = 0
    for i in range(n):
        total += i * i
    return total


def _optimized_workload(n: int) -> int:
    return sum(i * i for i in range(n))


def run_benchmark(n: int = 100_000) -> dict[str, float]:
    """Run baseline and optimized workloads and return throughput metrics."""

    storage = TimeSeriesStorage()
    monitor = PerformanceMonitor(
        storage,
        training_accuracy=1.0,
        degradation_threshold=0.1,
        cpu_threshold=90.0,
        memory_threshold=90.0,
        throughput_threshold=0.0,
        alert_handlers=[dashboard_alert(), email_alert("ops@example.com")],
    )

    process = psutil.Process()

    start = time.perf_counter()
    _baseline_workload(n)
    cpu = process.cpu_percent(interval=None)
    mem = process.memory_percent()
    monitor.log_resource_usage("benchmark", cpu, mem)
    monitor.log_task_completion("benchmark")
    baseline_time = time.perf_counter() - start

    start = time.perf_counter()
    _optimized_workload(n)
    cpu = process.cpu_percent(interval=None)
    mem = process.memory_percent()
    monitor.log_resource_usage("benchmark", cpu, mem)
    monitor.log_task_completion("benchmark")
    optimized_time = time.perf_counter() - start

    return {
        "throughput_before": 1.0 / baseline_time,
        "throughput_after": 1.0 / optimized_time,
    }


if __name__ == "__main__":
    result = run_benchmark()
    print(result)

