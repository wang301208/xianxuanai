import asyncio
import os
import sys
from pathlib import Path

import importlib.util
import types

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

if "events" not in sys.modules:
    events_stub = types.ModuleType("events")
    events_stub.EventBus = object
    sys.modules["events"] = events_stub

BACKEND_MONITORING = Path(ROOT, "backend", "monitoring")

if "backend" not in sys.modules:
    backend_pkg = types.ModuleType("backend")
    backend_pkg.__path__ = [str(Path(ROOT, "backend"))]  # type: ignore[attr-defined]
    sys.modules["backend"] = backend_pkg

if "backend.monitoring" not in sys.modules:
    monitoring_pkg = types.ModuleType("backend.monitoring")
    monitoring_pkg.__path__ = [str(BACKEND_MONITORING)]  # type: ignore[attr-defined]
    sys.modules["backend.monitoring"] = monitoring_pkg


def _load(name: str):
    module_path = BACKEND_MONITORING / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"backend.monitoring.{name}", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module  # type: ignore[attr-defined]
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


GlobalWorkspace = _load("global_workspace").GlobalWorkspace  # type: ignore[attr-defined]
ResourceScheduler = _load("resource_scheduler").ResourceScheduler  # type: ignore[attr-defined]


class DummyEventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list] = {}

    def publish(self, topic: str, event: dict) -> None:
        handlers = list(self._handlers.get(topic, []))
        for handler in handlers:
            result = handler(event)
            if asyncio.iscoroutine(result):
                asyncio.run(result)

    def subscribe(self, topic, handler):
        self._handlers.setdefault(topic, []).append(handler)

        def cancel() -> None:
            handlers = self._handlers.get(topic, [])
            if handler in handlers:
                handlers.remove(handler)
                if not handlers:
                    self._handlers.pop(topic, None)

        return cancel


class StubScheduler:
    def __init__(self) -> None:
        self.weights: dict[str, float] = {}

    def set_weights(self, **weights: float) -> None:
        self.weights.update(weights)


@pytest.fixture()
def event_bus() -> DummyEventBus:
    return DummyEventBus()


@pytest.fixture()
def scheduler() -> StubScheduler:
    return StubScheduler()


def test_threshold_scales_with_system_load(event_bus: DummyEventBus, scheduler: StubScheduler) -> None:
    workspace = GlobalWorkspace()
    controller = ResourceScheduler(
        workspace,
        event_bus,
        scheduler=scheduler,
        attention_bounds=(0.2, 0.8),
        smoothing=1.0,
    )

    try:
        event_bus.publish("agent.resource", {"agent": "alpha", "cpu": 80.0, "memory": 55.0})
        assert workspace.attention_threshold() == pytest.approx(0.68, abs=1e-2)
    finally:
        controller.close()


def test_alert_event_lowers_threshold(event_bus: DummyEventBus, scheduler: StubScheduler) -> None:
    workspace = GlobalWorkspace()
    controller = ResourceScheduler(
        workspace,
        event_bus,
        scheduler=scheduler,
        attention_bounds=(0.2, 0.8),
        smoothing=1.0,
    )

    try:
        event_bus.publish("agent.resource", {"agent": "alpha", "cpu": 80.0, "memory": 55.0})
        assert workspace.attention_threshold() == pytest.approx(0.68, abs=1e-2)

        event_bus.publish("environment.alert", {"severity": 0.9})
        assert workspace.attention_threshold() < 0.3
    finally:
        controller.close()


def test_backlog_updates_scheduler_weights(event_bus: DummyEventBus, scheduler: StubScheduler) -> None:
    workspace = GlobalWorkspace()
    controller = ResourceScheduler(
        workspace,
        event_bus,
        scheduler=scheduler,
        attention_bounds=(0.2, 0.8),
        smoothing=1.0,
        backlog_target=5,
    )

    try:
        controller.update_backlog(10)
        assert scheduler.weights["tasks"] > scheduler.weights["cpu"]
        assert scheduler.weights["tasks"] == pytest.approx(3.0, abs=1e-6)
    finally:
        controller.close()


def test_module_interval_adjusts_with_load_and_alert(event_bus: DummyEventBus, scheduler: StubScheduler) -> None:
    workspace = GlobalWorkspace()
    controller = ResourceScheduler(
        workspace,
        event_bus,
        scheduler=scheduler,
        attention_bounds=(0.2, 0.8),
        smoothing=1.0,
    )
    intervals: list[float] = []

    def adjust(value: float) -> None:
        intervals.append(value)

    controller.register_module(
        "sensor",
        adjust,
        base_interval=30.0,
        min_interval=5.0,
        max_interval=120.0,
        slowdown_factor=2.0,
        boost_factor=2.0,
    )

    try:
        event_bus.publish("agent.resource", {"agent": "alpha", "cpu": 70.0, "memory": 40.0})
        assert intervals
        assert intervals[-1] > 30.0

        previous = intervals[-1]
        event_bus.publish("environment.alert", {"severity": 1.0})
        assert intervals[-1] < previous
    finally:
        controller.close()
