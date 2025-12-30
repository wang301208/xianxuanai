"""Regression tests for AgentLifecycleManager workspace cleanup."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BACKEND_MONITORING = ROOT / "backend" / "monitoring"
BACKEND_EXECUTION = ROOT / "backend" / "execution"


def _load_monitoring_module(name: str):
    module_path = BACKEND_MONITORING / f"{name}.py"
    spec = importlib.util.spec_from_file_location(
        f"backend.monitoring.{name}", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module  # type: ignore[index]
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


_global_workspace_module = _load_monitoring_module("global_workspace")

if "backend.execution" not in sys.modules:
    execution_pkg = ModuleType("backend.execution")
    execution_pkg.__path__ = [str(BACKEND_EXECUTION)]  # type: ignore[attr-defined]
    sys.modules["backend.execution"] = execution_pkg


def _stub_module(name: str, **attrs: Any) -> None:
    module = ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    sys.modules[name] = module


_stub_module("backend.execution.scheduler", Scheduler=type("Scheduler", (), {}))
_stub_module("backend.execution.goal_generator", GoalGenerator=type("GoalGenerator", (), {"listener": None}))
_stub_module("backend.execution.adaptive_controller", AdaptiveResourceController=type("AdaptiveResourceController", (), {"shutdown": lambda self: None}))

class _TaskHandle:  # pragma: no cover - stub
    pass


class _TaskManagerStub:  # pragma: no cover - stub
    def __init__(self, *_, **__):
        pass

    def configure_device(self, *_: Any, **__: Any) -> None:
        pass

    def start(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


class _TaskPriority:  # pragma: no cover - stub
    NORMAL = "normal"


_stub_module(
    "backend.execution.task_manager",
    TaskHandle=_TaskHandle,
    TaskManager=_TaskManagerStub,
    TaskPriority=_TaskPriority,
)

_stub_module("backend.execution.planner", Planner=type("Planner", (), {}))
_stub_module("backend.execution.conductor", AgentConductor=type("AgentConductor", (), {}))
_stub_module("world_model", WorldModel=type("WorldModel", (), {}))
_stub_module("self_model", SelfModel=type("SelfModel", (), {}))
_stub_module("capability.runtime_loader", RuntimeModuleManager=type("RuntimeModuleManager", (), {}))
_stub_module(
    "knowledge",
    KnowledgeConsolidator=type("KnowledgeConsolidator", (), {}),
    MemoryRouter=type("MemoryRouter", (), {"add_observation": lambda self, *_, **__: None}),
)
_stub_module(
    "modules.knowledge",
    KnowledgeUpdatePipeline=type("KnowledgeUpdatePipeline", (), {"process_task_event": lambda self, *_, **__: None}),
    RuntimeKnowledgeImporter=type("RuntimeKnowledgeImporter", (), {}),
)
_stub_module("modules.environment.simulator", GridWorldEnvironment=type("GridWorldEnvironment", (), {}))
_stub_module(
    "modules.environment.loop",
    ActionPerceptionLoop=type("ActionPerceptionLoop", (), {"reset_environment": lambda self: None}),
)

if "capability" not in sys.modules:
    sys.modules["capability"] = ModuleType("capability")

if not hasattr(sys.modules["capability"], "refresh_skills_from_directory"):
    sys.modules["capability"].refresh_skills_from_directory = lambda *_, **__: None  # type: ignore[attr-defined]
    sys.modules["capability"].get_skill_registry = lambda: SimpleNamespace(list_specs=lambda: [])  # type: ignore[attr-defined]

if "agent_factory" not in sys.modules:
    agent_factory_stub = ModuleType("agent_factory")

    def _create_agent_from_blueprint(*_: Any, **__: Any) -> SimpleNamespace:
        return SimpleNamespace(workspace_keys=tuple())

    agent_factory_stub.create_agent_from_blueprint = _create_agent_from_blueprint  # type: ignore[attr-defined]
    sys.modules["agent_factory"] = agent_factory_stub

if "autogpt" not in sys.modules:
    sys.modules["autogpt"] = ModuleType("autogpt")

if "autogpt.config" not in sys.modules:
    config_stub = ModuleType("autogpt.config")

    class _Config:  # pragma: no cover - minimal stub
        prompt_settings_file = ""
        brain_backend = ""

    config_stub.Config = _Config  # type: ignore[attr-defined]
    sys.modules["autogpt.config"] = config_stub
    setattr(sys.modules["autogpt"], "config", config_stub)

if "autogpt.core" not in sys.modules:
    sys.modules["autogpt.core"] = ModuleType("autogpt.core")

if "autogpt.core.resource" not in sys.modules:
    sys.modules["autogpt.core.resource"] = ModuleType("autogpt.core.resource")

if "autogpt.core.resource.model_providers" not in sys.modules:
    model_stub = ModuleType("autogpt.core.resource.model_providers")

    class _ChatModelProvider:  # pragma: no cover - stub class
        pass

    model_stub.ChatModelProvider = _ChatModelProvider  # type: ignore[attr-defined]
    sys.modules["autogpt.core.resource.model_providers"] = model_stub
    setattr(sys.modules["autogpt.core.resource"], "model_providers", model_stub)

if "autogpt.file_storage" not in sys.modules:
    sys.modules["autogpt.file_storage"] = ModuleType("autogpt.file_storage")

if "autogpt.file_storage.base" not in sys.modules:
    storage_stub = ModuleType("autogpt.file_storage.base")

    class _FileStorage:  # pragma: no cover - stub class
        pass

    storage_stub.FileStorage = _FileStorage  # type: ignore[attr-defined]
    sys.modules["autogpt.file_storage.base"] = storage_stub
    setattr(sys.modules["autogpt.file_storage"], "base", storage_stub)

if "autogpt.agents" not in sys.modules:
    sys.modules["autogpt.agents"] = ModuleType("autogpt.agents")

if "autogpt.agents.agent" not in sys.modules:
    agent_stub = ModuleType("autogpt.agents.agent")

    class _Agent:  # pragma: no cover - stub class
        pass

    agent_stub.Agent = _Agent  # type: ignore[attr-defined]
    sys.modules["autogpt.agents.agent"] = agent_stub
    setattr(sys.modules["autogpt.agents"], "agent", agent_stub)

if "events" not in sys.modules:
    events_stub = ModuleType("events")

    class _EventBus:  # pragma: no cover - stub
        def subscribe(self, *_: Any, **__: Any):
            return lambda: None

        def publish(self, *_: Any, **__: Any) -> None:
            pass

    events_stub.EventBus = _EventBus  # type: ignore[attr-defined]
    events_stub.AsyncEventBus = _EventBus  # type: ignore[attr-defined]
    events_stub.InMemoryEventBus = _EventBus  # type: ignore[attr-defined]
    events_stub.create_event_bus = lambda *_, **__: _EventBus()
    events_stub.publish = lambda *_, **__: None
    events_stub.subscribe = lambda *_, **__: (lambda: None)
    events_stub.unsubscribe = lambda *_, **__: None
    sys.modules["events"] = events_stub

if "org_charter" not in sys.modules:
    sys.modules["org_charter"] = ModuleType("org_charter")

if "org_charter.watchdog" not in sys.modules:
    watchdog_stub = ModuleType("org_charter.watchdog")

    class _DummyWatcher:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    watchdog_stub.BlueprintWatcher = _DummyWatcher  # type: ignore[attr-defined]
    sys.modules["org_charter.watchdog"] = watchdog_stub
    setattr(sys.modules["org_charter"], "watchdog", watchdog_stub)

if "psutil" not in sys.modules:
    class _DummyProcess:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def cpu_percent(self, *_: Any, **__: Any) -> float:
            return 0.0

        def memory_percent(self) -> float:
            return 0.0

    psutil_stub = ModuleType("psutil")
    psutil_stub.Process = _DummyProcess  # type: ignore[attr-defined]
    psutil_stub.NoSuchProcess = Exception  # type: ignore[attr-defined]
    psutil_stub.AccessDenied = Exception  # type: ignore[attr-defined]
    sys.modules["psutil"] = psutil_stub

if "monitoring" not in sys.modules:
    _gw = _global_workspace_module.global_workspace

    monitoring_stub = ModuleType("monitoring")

    class _Placeholder:  # pragma: no cover - minimal shim for imports
        pass

    monitoring_stub.ResourceScheduler = _Placeholder  # type: ignore[attr-defined]
    monitoring_stub.SystemMetricsCollector = _Placeholder  # type: ignore[attr-defined]
    monitoring_stub.global_workspace = _gw  # type: ignore[attr-defined]
    sys.modules["monitoring"] = monitoring_stub

if "capability.skill_library" not in sys.modules:
    skill_lib = ModuleType("capability.skill_library")
    skill_lib.SkillLibrary = object  # type: ignore[attr-defined]
    sys.modules["capability.skill_library"] = skill_lib
    setattr(sys.modules["capability"], "skill_library", skill_lib)

if "structlog" not in sys.modules:
    structlog_stub = ModuleType("structlog")
    structlog_stub.get_logger = lambda *_, **__: SimpleNamespace()
    sys.modules["structlog"] = structlog_stub

if "common" not in sys.modules:
    sys.modules["common"] = ModuleType("common")

common_module = sys.modules["common"]
if not hasattr(common_module, "AutoGPTException"):
    class _AutoGPTException(Exception):
        pass

    def _log_and_format_exception(exc: Exception) -> Dict[str, Any]:
        return {"error": str(exc)}

    common_module.AutoGPTException = _AutoGPTException  # type: ignore[attr-defined]
    common_module.log_and_format_exception = _log_and_format_exception  # type: ignore[attr-defined]

if "common.async_utils" not in sys.modules:
    async_utils = ModuleType("common.async_utils")
    async_utils.run_async = lambda coro: coro
    sys.modules["common.async_utils"] = async_utils
    setattr(sys.modules["common"], "async_utils", async_utils)

if "aiofiles" not in sys.modules:
    class _DummyFile:
        async def read(self) -> str:
            return ""

    class _DummyContext:
        async def __aenter__(self) -> _DummyFile:  # type: ignore[name-defined]
            return _DummyFile()

        async def __aexit__(self, *_: Any) -> None:
            return None

    aiofiles_stub = ModuleType("aiofiles")
    aiofiles_stub.open = lambda *_, **__: _DummyContext()
    sys.modules["aiofiles"] = aiofiles_stub

import backend.execution.manager as manager_module
from backend.monitoring.global_workspace import global_workspace


class DummyEventBus:
    def __init__(self) -> None:
        self._subscriptions: Dict[str, List[Callable[[Any], Any]]] = {}

    def subscribe(self, topic: str, handler: Callable[[Any], Any]) -> Callable[[], None]:
        self._subscriptions.setdefault(topic, []).append(handler)

        def _cancel() -> None:
            handlers = self._subscriptions.get(topic)
            if handlers and handler in handlers:
                handlers.remove(handler)

        return _cancel

    def publish(self, topic: str, payload: Any) -> None:  # pragma: no cover - not used
        pass


class DummyWatcher:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class DummyMetrics:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def register(self, *_: Any, **__: Any) -> None:
        pass

    def unregister(self, *_: Any, **__: Any) -> None:
        pass


class DummyResourceScheduler:
    def __init__(self, *_, **__):
        pass

    def register_module(self, *_, **__):
        pass

    def update_backlog(self, *_: Any, **__: Any) -> None:
        pass

    def close(self) -> None:
        pass


class DummyWorldModel:
    def __init__(self, *_, **__):
        pass


class DummySelfModel:
    def __init__(self, *_, **__):
        pass


class DummyPlanner:
    def __init__(self, *_, **__):
        pass


class DummyGoalGenerator:
    def __init__(self, *_, **__):
        self.listener = None

    def stop(self) -> None:
        pass


class DummyTaskManager:
    def __init__(self, *_, **__):
        pass

    def configure_device(self, *_: Any, **__: Any) -> None:
        pass

    def start(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


class DummyAdaptiveController:
    def __init__(self, *_, **__):
        pass

    def shutdown(self) -> None:
        pass


class DummyKnowledgeConsolidator:
    def __init__(self, *_, **__):
        pass


class DummyMemoryRouter:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def add_observation(self, *_: Any, **__: Any) -> None:
        pass


class DummyKnowledgeImporter:
    def __init__(self, *_, **__):
        pass


class DummyKnowledgePipeline:
    def __init__(self, *_, **__):
        pass

    def process_task_event(self, *_: Any, **__: Any) -> None:
        pass


class DummyEnvironment:
    pass


class DummyEnvironmentLoop:
    def __init__(self, *_, **__):
        pass

    def reset_environment(self) -> None:
        pass


class DummyModuleManager:
    def __init__(self, *_, **__):
        pass


class DummyConductor:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def register_agent(self, *_: Any, **__: Any) -> None:
        pass

    def unregister_agent(self, *_: Any, **__: Any) -> None:
        pass

    def close(self) -> None:
        pass


def test_manager_stop_unregisters_global_workspace(monkeypatch):
    # Ensure any stale keys from prior runs are cleared
    for key in (
        "async.executor",
        "async.loop",
        "test-agent",
        "test-agent.command_registry",
        "test-agent.llm_provider",
    ):
        global_workspace.unregister_module(key)

    monkeypatch.setattr(manager_module, "BlueprintWatcher", DummyWatcher)
    monkeypatch.setattr(manager_module, "SystemMetricsCollector", DummyMetrics)
    monkeypatch.setattr(manager_module, "ResourceScheduler", DummyResourceScheduler)
    monkeypatch.setattr(manager_module, "WorldModel", DummyWorldModel)
    monkeypatch.setattr(manager_module, "SelfModel", DummySelfModel)
    monkeypatch.setattr(manager_module, "Planner", DummyPlanner)
    monkeypatch.setattr(manager_module, "GoalGenerator", DummyGoalGenerator)
    monkeypatch.setattr(manager_module, "TaskManager", DummyTaskManager)
    monkeypatch.setattr(manager_module, "AdaptiveResourceController", DummyAdaptiveController)
    monkeypatch.setattr(manager_module, "KnowledgeConsolidator", DummyKnowledgeConsolidator)
    monkeypatch.setattr(manager_module, "MemoryRouter", DummyMemoryRouter)
    monkeypatch.setattr(manager_module, "RuntimeKnowledgeImporter", DummyKnowledgeImporter)
    monkeypatch.setattr(manager_module, "KnowledgeUpdatePipeline", DummyKnowledgePipeline)
    monkeypatch.setattr(manager_module, "GridWorldEnvironment", DummyEnvironment)
    monkeypatch.setattr(manager_module, "ActionPerceptionLoop", DummyEnvironmentLoop)
    monkeypatch.setattr(manager_module, "RuntimeModuleManager", DummyModuleManager)
    monkeypatch.setattr(manager_module, "AgentConductor", DummyConductor)

    config = SimpleNamespace()
    event_bus = DummyEventBus()
    manager = manager_module.AgentLifecycleManager(
        config=config,
        llm_provider=object(),
        file_storage=object(),
        event_bus=event_bus,
    )

    assert global_workspace.state("async.executor") is not None
    assert global_workspace.state("async.loop") is not None

    agent = SimpleNamespace(
        workspace_keys=(
            "test-agent",
            "test-agent.command_registry",
            "test-agent.llm_provider",
        )
    )
    for key in agent.workspace_keys:
        global_workspace.register_module(key, object())
        global_workspace.broadcast(key, {"value": key})
    manager._agents["test-agent"] = agent

    manager.stop()

    assert global_workspace.state("async.executor") is None
    assert global_workspace.state("async.loop") is None

    modules = getattr(global_workspace, "_modules")
    for key in agent.workspace_keys:
        assert key not in modules
        assert global_workspace.state(key) is None
