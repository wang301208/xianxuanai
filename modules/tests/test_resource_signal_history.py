import os
import sys

sys.path.insert(0, os.path.abspath(os.getcwd()))

from events import InMemoryEventBus

from modules.environment.registry import get_hardware_registry
from modules.environment.resource_history import ResourceSignalHistory


def test_resource_signal_history_updates_registry_and_window() -> None:
    registry = get_hardware_registry()
    registry.unregister("w1")

    current = [0.0]

    def now() -> float:
        return float(current[0])

    bus = InMemoryEventBus()
    history = ResourceSignalHistory(event_bus=bus, window_seconds=10.0, time_source=now)
    try:
        bus.publish(
            "resource.signal",
            {"worker_id": "w1", "resource_signal": {"cpu_percent": 10.0, "memory_percent": 20.0}},
        )
        bus.join()

        current[0] = 5.0
        bus.publish(
            "resource.signal",
            {"worker_id": "w1", "resource_signal": {"cpu_percent": 30.0, "memory_percent": 40.0}},
        )
        bus.join()

        latest = registry.get("w1") or {}
        assert (latest.get("metrics") or {}).get("cpu_percent") == 30.0

        means = history.rolling_mean("w1", keys=("cpu_percent", "memory_percent"))
        assert means["cpu_percent"] == 20.0
        assert means["memory_percent"] == 30.0

        current[0] = 25.0
        bus.publish(
            "resource.signal",
            {"worker_id": "w1", "resource_signal": {"cpu_percent": 50.0, "memory_percent": 60.0}},
        )
        bus.join()

        samples = history.worker_samples("w1")
        assert len(samples) == 1
        means = history.rolling_mean("w1", keys=("cpu_percent", "memory_percent"))
        assert means["cpu_percent"] == 50.0
        assert means["memory_percent"] == 60.0
    finally:
        history.close()
        registry.unregister("w1")


def test_resource_signal_history_detects_sustained_high() -> None:
    registry = get_hardware_registry()
    registry.unregister("hot")

    current = [0.0]

    def now() -> float:
        return float(current[0])

    bus = InMemoryEventBus()
    history = ResourceSignalHistory(event_bus=bus, window_seconds=300.0, time_source=now)
    try:
        bus.publish(
            "resource.signal",
            {"worker_id": "hot", "resource_signal": {"cpu_percent": 95.0, "memory_percent": 10.0}},
        )
        bus.join()

        current[0] = 10.0
        bus.publish(
            "resource.signal",
            {"worker_id": "hot", "resource_signal": {"cpu_percent": 96.0, "memory_percent": 10.0}},
        )
        bus.join()

        current[0] = 30.0
        bus.publish(
            "resource.signal",
            {"worker_id": "hot", "resource_signal": {"cpu_percent": 94.0, "memory_percent": 10.0}},
        )
        bus.join()

        assert history.sustained_high("hot", cpu_threshold=90.0, memory_threshold=85.0, sustain_seconds=30.0)
    finally:
        history.close()
        registry.unregister("hot")
