import sys, os, time
from pathlib import Path
import types

sys.path.insert(0, os.path.abspath(os.getcwd()))

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


sys.modules["org_charter.watchdog"] = types.SimpleNamespace(
    BlueprintWatcher=_StubWatcher
)
sys.modules["monitoring"] = types.SimpleNamespace(
    SystemMetricsCollector=lambda bus: types.SimpleNamespace(
        start=lambda: None,
        stop=lambda: None,
        register=lambda name, pid: None,
        unregister=lambda name: None,
    )
)

import execution.manager as manager
from execution.manager import AgentLifecycleManager, AgentState
from events import InMemoryEventBus


class DummyMetrics:
    def __init__(self, bus):
        self.registered = []
        self.unregistered = []

    def start(self):
        pass

    def stop(self):
        pass

    def register(self, name, pid):
        self.registered.append(name)

    def unregister(self, name):
        self.unregistered.append(name)


class DummyWatcher:
    def __init__(self, cb):
        pass

    def start(self):
        pass

    def stop(self):
        pass


class DummyScheduler:
    def __init__(self):
        self.removed = []

    def set_task_callback(self, cb):
        pass

    def remove_agent(self, name):
        self.removed.append(name)

    def add_agent(self, name):
        pass

    def update_agent(self, name, cpu, mem):
        pass


def test_cleanup_removes_long_sleeping_agents(monkeypatch):
    events = []
    bus = InMemoryEventBus()
    bus.subscribe("agent.state", lambda e: events.append(e))

    monkeypatch.setattr(manager, "SystemMetricsCollector", DummyMetrics)
    monkeypatch.setattr(manager, "BlueprintWatcher", DummyWatcher)

    sched = DummyScheduler()
    mgr = AgentLifecycleManager(
        config=types.SimpleNamespace(),
        llm_provider=object(),
        file_storage=object(),
        event_bus=bus,
        scheduler=sched,
        sleep_timeout=0.1,
    )

    mgr._states["test"] = AgentState.SLEEPING
    mgr._heartbeats["test"] = time.time() - 1.0
    mgr._agents["test"] = object()
    mgr._resources["test"] = {}
    mgr._paths["test"] = Path("dummy")

    mgr._cleanup_sleeping_agents(time.time())
    bus.join()

    assert "test" not in mgr._states
    assert "test" not in mgr._agents
    assert "test" not in mgr._resources
    assert "test" not in mgr._paths
    assert "test" not in mgr._heartbeats
    assert sched.removed == ["test"]
    assert mgr._metrics.unregistered == ["test"]
    assert any(e["agent"] == "test" and e["state"] == "terminated" for e in events)

    mgr.stop()

