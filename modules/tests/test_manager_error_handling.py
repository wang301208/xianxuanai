import sys, os, types
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.getcwd()))

# Stub heavy dependencies before importing manager
sys.modules["agent_factory"] = types.SimpleNamespace(
    create_agent_from_blueprint=lambda *a, **k: object()
)
sys.modules["autogpt"] = types.SimpleNamespace()
sys.modules["autogpt.config"] = types.SimpleNamespace(Config=object)
sys.modules["autogpt.core.resource.model_providers"] = types.SimpleNamespace(
    ChatModelProvider=object
)
sys.modules["autogpt.file_storage.base"] = types.SimpleNamespace(FileStorage=object)
sys.modules["autogpt.agents.agent"] = types.SimpleNamespace(Agent=object)


class _StubWatcher:
    def __init__(self, cb):
        pass

    def start(self):
        pass

    def stop(self):
        pass


sys.modules["org_charter.watchdog"] = types.SimpleNamespace(BlueprintWatcher=_StubWatcher)

from events import InMemoryEventBus
from common import AutoGPTException
import execution.manager as manager
from execution.manager import AgentLifecycleManager


class DummyMetrics:
    def __init__(self, bus):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def register(self, name, pid):
        pass

    def unregister(self, name):
        pass


class DummyWatcher:
    def __init__(self, cb):
        pass

    def start(self):
        pass

    def stop(self):
        pass


def _make_manager(bus, monkeypatch):
    monkeypatch.setattr(manager, "SystemMetricsCollector", DummyMetrics)
    monkeypatch.setattr(manager, "BlueprintWatcher", DummyWatcher)
    return AgentLifecycleManager(
        config=types.SimpleNamespace(),
        llm_provider=object(),
        file_storage=object(),
        event_bus=bus,
    )


def test_on_blueprint_change_spawn(monkeypatch):
    events = []
    bus = InMemoryEventBus()
    bus.subscribe("agent.lifecycle", lambda e: events.append(e))
    monkeypatch.setattr(manager, "create_agent_from_blueprint", lambda *a, **k: object())
    mgr = _make_manager(bus, monkeypatch)
    path = Path("agent_v1.json")
    mgr._on_blueprint_change(path)
    bus.join()
    assert any(e["action"] == "spawned" for e in events)
    mgr.stop()


def test_on_blueprint_change_error(monkeypatch, caplog):
    events = []
    bus = InMemoryEventBus()
    bus.subscribe("agent.lifecycle", lambda e: events.append(e))

    def raising(*a, **k):
        raise AutoGPTException("boom")

    monkeypatch.setattr(manager, "create_agent_from_blueprint", raising)
    mgr = _make_manager(bus, monkeypatch)
    path = Path("agent_v1.json")
    with caplog.at_level("ERROR"):
        mgr._on_blueprint_change(path)
    bus.join()
    assert events[0]["action"] == "failed"
    assert events[0]["error_type"] == "AutoGPTException"
    assert any("AutoGPTException" in r.message for r in caplog.records)
    mgr.stop()
