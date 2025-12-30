# -*- coding: utf-8 -*-

from BrainSimulationSystem.core.performance_monitor import PerformanceMonitor


def test_performance_monitor_recommendations():
    monitor = PerformanceMonitor(
        {
            "window": 5,
            "max_region_time": 0.02,
            "min_region_time": 0.005,
            "max_global_time": 0.05,
            "evaluation_interval": 1,
        }
    )

    for _ in range(5):
        monitor.record_global_step(0.06)
        monitor.record_region_step("V1", 0.03)
        monitor.record_region_step("PFC", 0.002)

    actions = monitor.generate_actions()

    assert actions["gpu"]["enable"] == "true"
    assert actions["region_modes"]["V1"] == "macro"
    assert actions["region_modes"]["PFC"] == "micro"
