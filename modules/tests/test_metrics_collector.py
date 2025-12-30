import sys, os
sys.path.insert(0, os.path.abspath(os.getcwd()))

from pathlib import Path

from events import InMemoryEventBus
from monitoring.metrics_collector import MetricsCollector


def test_metrics_collector_stores_events(tmp_path: Path) -> None:
    bus = InMemoryEventBus()
    db_path = tmp_path / "metrics.db"
    collector = MetricsCollector(bus, db_path)
    bus.publish("agent.lifecycle", {"agent": "test", "action": "spawned"})
    bus.publish("agent.resource", {"agent": "test", "cpu": 10.0, "memory": 20.0})
    bus.join()
    events = collector.storage.events("agent.resource")
    assert events[0]["cpu"] == 10.0
    collector.close()
