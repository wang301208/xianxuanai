import sys
import types
from importlib import util
from pathlib import Path
from typing import Any, Dict, List

from events import InMemoryEventBus


ROOT = Path(__file__).resolve().parents[2]

backend_pkg = types.ModuleType("backend")
backend_pkg.__path__ = [str(ROOT / "backend")]
sys.modules.setdefault("backend", backend_pkg)

execution_pkg = types.ModuleType("backend.execution")
execution_pkg.__path__ = [str(ROOT / "backend" / "execution")]
sys.modules.setdefault("backend.execution", execution_pkg)

MODULE_PATH = ROOT / "backend" / "execution" / "module_acquisition.py"
spec = util.spec_from_file_location("backend.execution.module_acquisition", MODULE_PATH)
module = util.module_from_spec(spec)
assert spec and spec.loader
sys.modules.setdefault("backend.execution.module_acquisition", module)
spec.loader.exec_module(module)

ModuleAcquisitionManager = module.ModuleAcquisitionManager


def _stub_skill_registry(monkeypatch) -> None:
    class _StubSpec:
        def __init__(self, name: str, description: str) -> None:
            self.name = name
            self.description = description
            self.enabled = True

    class _StubRegistry:
        def list_specs(self):
            return [_StubSpec("known_skill", "A known skill used for tests.")]

    import backend.capability.skill_registry as skill_registry

    monkeypatch.setattr(skill_registry, "get_skill_registry", lambda: _StubRegistry())


def test_find_internal_surfaces_capability_modules(monkeypatch) -> None:
    _stub_skill_registry(monkeypatch)

    import backend.capability.module_registry as module_registry

    monkeypatch.setattr(module_registry, "_REGISTRY", {"demo_module": lambda: {"ok": True}})

    bus = InMemoryEventBus()
    manager = ModuleAcquisitionManager(
        enabled=True,
        event_bus=bus,  # type: ignore[arg-type]
        module_manager=None,
        task_manager=None,
        search_docs=False,
        search_web=False,
        cooldown_secs=0.0,
    )

    matches = manager.find_internal("demo_module", limit=3)
    assert matches
    assert matches[0].kind == "capability"
    assert matches[0].name == "demo_module"
    assert matches[0].score >= 0.99


def test_request_publishes_suggestion_for_missing_module(monkeypatch) -> None:
    _stub_skill_registry(monkeypatch)

    import backend.capability.module_registry as module_registry

    monkeypatch.setattr(module_registry, "_REGISTRY", {"known": lambda: object()})

    events: List[Dict[str, Any]] = []
    bus = InMemoryEventBus()
    bus.subscribe("module.acquisition.suggested", lambda e: events.append(e))

    manager = ModuleAcquisitionManager(
        enabled=True,
        event_bus=bus,  # type: ignore[arg-type]
        module_manager=None,
        task_manager=None,
        search_docs=False,
        search_web=False,
        cooldown_secs=0.0,
    )

    missing = manager.request_for_tasks(["unknown_task"], goal="do something", reason="unit_test")
    assert missing == ["unknown_task"]

    bus.join()
    assert events
    suggestion = events[0]["suggestion"]
    assert suggestion["query"] == "unknown_task"


def test_request_loads_exact_match_module(monkeypatch) -> None:
    _stub_skill_registry(monkeypatch)

    import backend.capability.module_registry as module_registry
    from backend.capability.runtime_loader import RuntimeModuleManager

    monkeypatch.setattr(module_registry, "_REGISTRY", {"hello": lambda: {"value": 1}})

    loaded_events: List[Dict[str, Any]] = []
    bus = InMemoryEventBus()
    bus.subscribe("module.acquisition.loaded", lambda e: loaded_events.append(e))

    loader = RuntimeModuleManager(None)
    manager = ModuleAcquisitionManager(
        enabled=True,
        event_bus=bus,  # type: ignore[arg-type]
        module_manager=loader,
        task_manager=None,
        search_docs=False,
        search_web=False,
        cooldown_secs=0.0,
    )

    manager.request("hello", context={"reason": "unit_test"})

    bus.join()
    assert loaded_events
    assert loaded_events[0]["module"] == "hello"

