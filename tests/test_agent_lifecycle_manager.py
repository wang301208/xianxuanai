import sys, os, types, time, importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "modules"))

# Stub dependencies required by events module
sys.modules.setdefault("autogpts", types.ModuleType("autogpts"))
sys.modules.setdefault("third_party.autogpt", types.ModuleType("third_party.autogpt"))
sys.modules.setdefault(
    "third_party.autogpt.autogpt", types.ModuleType("third_party.autogpt.autogpt")
)
core_mod = types.ModuleType("third_party.autogpt.autogpt.core")
errors_mod = types.ModuleType("third_party.autogpt.autogpt.core.errors")
class _AutoGPTError(Exception):
    pass
errors_mod.AutoGPTError = _AutoGPTError
core_mod.errors = errors_mod
logging_mod = types.ModuleType("third_party.autogpt.autogpt.core.logging")
logging_mod.handle_exception = lambda *a, **k: None
sys.modules["third_party.autogpt.autogpt.core.logging"] = logging_mod
sys.modules["third_party.autogpt.autogpt.core"] = core_mod
sys.modules["third_party.autogpt.autogpt.core.errors"] = errors_mod

from events import InMemoryEventBus


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


class DummyWorkspace:
    def broadcast(self, *a, **k):
        pass


def _load_manager(monkeypatch):
    monkeypatch.setitem(sys.modules, "agent_factory", types.SimpleNamespace(
        create_agent_from_blueprint=lambda *a, **k: object()
    ))
    monkeypatch.setitem(sys.modules, "autogpt", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "autogpt.config", types.SimpleNamespace(Config=object))
    monkeypatch.setitem(
        sys.modules,
        "autogpt.core.resource.model_providers",
        types.SimpleNamespace(ChatModelProvider=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "autogpt.file_storage.base",
        types.SimpleNamespace(FileStorage=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "autogpt.agents.agent",
        types.SimpleNamespace(Agent=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "monitoring",
        types.SimpleNamespace(
            SystemMetricsCollector=DummyMetrics,
            global_workspace=DummyWorkspace(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "org_charter.watchdog",
        types.SimpleNamespace(BlueprintWatcher=DummyWatcher),
    )
    monkeypatch.setitem(
        sys.modules,
        "common",
        types.SimpleNamespace(
            AutoGPTException=Exception,
            log_and_format_exception=lambda e: {
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
        ),
    )

    class _DummyScheduler:
        def set_task_callback(self, cb):
            pass
        def add_agent(self, name):
            pass
        def remove_agent(self, name):
            pass
        def update_agent(self, name, cpu, memory):
            pass

    class _DummyPlanner:
        def decompose(self, goal, source="auto"):
            pass

    class _DummyGoalGenerator:
        def __init__(self, *a, **k):
            pass

        def generate(self):
            return None

        def stop(self):
            pass

    class _DummyWorldModel:
        def predict(self, resources):
            return {}

    class _DummySelfModel:
        def assess_state(self, data, env_pred, action):
            return {"cpu": 0.0, "memory": 0.0}, ""

    monkeypatch.setitem(
        sys.modules,
        "backend.execution.scheduler",
        types.SimpleNamespace(Scheduler=_DummyScheduler),
    )
    monkeypatch.setitem(
        sys.modules,
        "backend.execution.planner",
        types.SimpleNamespace(Planner=_DummyPlanner),
    )
    monkeypatch.setitem(
        sys.modules,
        "backend.execution.goal_generator",
        types.SimpleNamespace(GoalGenerator=_DummyGoalGenerator),
    )
    monkeypatch.setitem(
        sys.modules,
        "world_model",
        types.SimpleNamespace(WorldModel=_DummyWorldModel),
    )
    monkeypatch.setitem(
        sys.modules,
        "self_model",
        types.SimpleNamespace(SelfModel=_DummySelfModel),
    )
    class _DummyRuntimeModuleManager:
        def __init__(self, *a, **k):
            pass

        def update(self, tasks):
            pass

    monkeypatch.setitem(
        sys.modules,
        "capability.runtime_loader",
        types.SimpleNamespace(RuntimeModuleManager=_DummyRuntimeModuleManager),
    )
    class _DummyKnowledgeConsolidator:
        def __init__(self, *a, **k):
            pass

    class _DummyMemoryRouter:
        def __init__(self, *a, **k):
            pass

        def add_observation(self, *a, **k):
            pass

        def review(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "knowledge",
        types.SimpleNamespace(
            KnowledgeConsolidator=_DummyKnowledgeConsolidator,
            MemoryRouter=_DummyMemoryRouter,
        ),
    )

    # Create package structure for relative imports
    backend_pkg = types.ModuleType("backend")
    backend_pkg.__path__ = [str(ROOT / "backend")]
    sys.modules.setdefault("backend", backend_pkg)
    execution_pkg = types.ModuleType("backend.execution")
    execution_pkg.__path__ = [str(ROOT / "backend" / "execution")]
    sys.modules.setdefault("backend.execution", execution_pkg)

    spec = importlib.util.spec_from_file_location(
        "backend.execution.manager", ROOT / "backend" / "execution" / "manager.py"
    )
    manager = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(manager)
    return manager


def _make_manager(monkeypatch, bus):
    manager = _load_manager(monkeypatch)
    mgr = manager.AgentLifecycleManager(
        config=types.SimpleNamespace(),
        llm_provider=object(),
        file_storage=object(),
        event_bus=bus,
        sleep_timeout=0.1,
    )
    # stop background resource thread
    mgr._resource_stop.set()
    mgr._resource_thread.join()
    return mgr, manager.AgentState


def test_pause_resume_terminate(monkeypatch):
    events = []
    bus = InMemoryEventBus()
    bus.subscribe("agent.lifecycle", lambda e: events.append(e))
    mgr, AgentState = _make_manager(monkeypatch, bus)

    name = "agent1"
    mgr._agents[name] = object()
    mgr._states[name] = AgentState.RUNNING
    mgr._paths[name] = Path("agent_v1.json")

    mgr.pause_agent(name)
    bus.join()
    assert mgr._states[name] == AgentState.SLEEPING
    assert any(e["action"] == "paused" for e in events)

    mgr.resume_agent(name)
    bus.join()
    assert mgr._states[name] == AgentState.RUNNING
    assert any(e["action"] == "resumed" for e in events)

    mgr.terminate_agent(name)
    bus.join()
    assert mgr._states[name] == AgentState.TERMINATED
    assert any(e["action"] == "terminated" for e in events)
    mgr.stop()


def test_heartbeat_timeouts(monkeypatch):
    events = []
    bus = InMemoryEventBus()
    bus.subscribe("agent.lifecycle", lambda e: events.append(e))
    mgr, AgentState = _make_manager(monkeypatch, bus)

    mgr._heartbeat_timeout = 0.05
    mgr._sleep_timeout = 0.1

    name = "agent2"
    mgr._agents[name] = object()
    mgr._states[name] = AgentState.RUNNING
    mgr._paths[name] = Path("agent_v1.json")
    mgr._resources[name] = {}
    mgr._heartbeats[name] = time.time() - 1.0

    class RunOnce:
        def __init__(self):
            self.count = 0
        def wait(self, timeout):
            self.count += 1
            return self.count > 1
        def set(self):
            self.count = 2

    mgr._resource_stop = RunOnce()
    mgr._resource_manager()
    bus.join()
    assert mgr._states[name] == AgentState.SLEEPING
    assert any(e["action"] == "paused" for e in events)

    time.sleep(0.15)
    mgr._resource_stop = RunOnce()
    mgr._resource_manager()
    bus.join()
    assert any(e["action"] == "terminated" for e in events)
    assert mgr._states.get(name) == AgentState.TERMINATED or name not in mgr._states
    mgr.stop()
