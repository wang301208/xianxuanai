import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.brain.brain_manager import UnifiedBrainManager
from modules.brain.message_bus import (
    publish_neural_event,
    subscribe_to_brain_region,
    reset_message_bus,
)


class DummyLifecycleManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def pause_agent(self, name: str, reason: str | None = None) -> None:
        self.calls.append((name, reason))

    def resume_agent(self, name: str) -> None:  # pragma: no cover - unused
        self.calls.append(("resume", name))


class DummyModuleManager:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def load_module(self, name: str) -> None:
        self.events.append(("load", name))

    def unload_module(self, name: str) -> None:
        self.events.append(("unload", name))


class FailingLifecycleManager(DummyLifecycleManager):
    def __init__(self) -> None:
        super().__init__()
        self.failures = 0

    def pause_agent(self, name: str, reason: str | None = None) -> None:
        self.failures += 1
        raise RuntimeError("boom")


def setup_function() -> None:
    # Ensure a clean message bus for each test
    reset_message_bus()


def test_event_driven_coordination() -> None:
    lifecycle = DummyLifecycleManager()
    modules = DummyModuleManager()
    UnifiedBrainManager(lifecycle, modules)

    publish_neural_event(
        {"target": "brain_manager", "command": "pause_agent", "agent": "007", "reason": "rest"}
    )
    publish_neural_event(
        {"target": "brain_manager", "command": "load_module", "module": "vision"}
    )

    assert lifecycle.calls == [("007", "rest")]
    assert modules.events == [("load", "vision")]


def test_fault_isolation_and_notification() -> None:
    lifecycle = DummyLifecycleManager()
    modules = DummyModuleManager()
    UnifiedBrainManager(lifecycle, modules)

    notifications: list[dict] = []
    subscribe_to_brain_region("monitor", lambda e: notifications.append(e))

    publish_neural_event(
        {"target": "brain_manager", "command": "fault", "agent": "bob", "notify": "monitor"}
    )

    assert lifecycle.calls == [("bob", "fault")]
    assert notifications and notifications[-1]["status"] == "isolated"


def test_circuit_breaker_isolates_failures() -> None:
    lifecycle = FailingLifecycleManager()
    modules = DummyModuleManager()
    UnifiedBrainManager(lifecycle, modules)

    for _ in range(4):
        publish_neural_event(
            {"target": "brain_manager", "command": "pause_agent", "agent": "bad"}
        )

    # Circuit breaker should stop after three failures
    assert lifecycle.failures == 3
