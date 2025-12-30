import sys
import types
from importlib import util
from pathlib import Path
from typing import Any, Dict, List

from modules.events import InMemoryEventBus


ROOT = Path(__file__).resolve().parents[2]

backend_pkg = types.ModuleType("backend")
backend_pkg.__path__ = [str(ROOT / "backend")]
sys.modules.setdefault("backend", backend_pkg)

execution_pkg = types.ModuleType("backend.execution")
execution_pkg.__path__ = [str(ROOT / "backend" / "execution")]
sys.modules.setdefault("backend.execution", execution_pkg)

MODULE_PATH = ROOT / "backend" / "execution" / "module_lifecycle_manager.py"
spec = util.spec_from_file_location("backend.execution.module_lifecycle_manager", MODULE_PATH)
module = util.module_from_spec(spec)
assert spec and spec.loader
sys.modules.setdefault("backend.execution.module_lifecycle_manager", module)
spec.loader.exec_module(module)

ModuleLifecycleManager = module.ModuleLifecycleManager


def test_lifecycle_manager_suggests_unload_for_idle_loaded_module(tmp_path) -> None:
    from backend.capability import register_module
    from backend.capability.runtime_loader import RuntimeModuleManager

    register_module("cold_mod", lambda: {"ok": True})

    bus = InMemoryEventBus()
    lifecycle_events: List[Dict[str, Any]] = []
    bus.subscribe("module.lifecycle.suggest_unload", lambda e: lifecycle_events.append(e))

    mgr = RuntimeModuleManager(bus)  # publish module.loaded/unloaded events
    lifecycle = ModuleLifecycleManager(
        event_bus=bus,  # type: ignore[arg-type]
        module_manager=mgr,
        enabled=True,
        disabled_state_path=tmp_path / "disabled.json",
        eval_interval_secs=0.0,
        unload_idle_secs=0.0,
        disable_idle_secs=999999.0,
        auto_unload=False,
    )

    mgr.load("cold_mod")
    bus.join()
    bus.publish("learning.cycle_completed", {"ok": True, "stats": {}})
    bus.join()

    assert lifecycle_events
    assert lifecycle_events[0]["module"] == "cold_mod"

    lifecycle.close()


def test_lifecycle_manager_can_disable_module_via_event(tmp_path) -> None:
    from backend.capability import is_module_enabled, register_module
    from backend.capability.runtime_loader import RuntimeModuleManager

    register_module("rare_mod", lambda: {"ok": True})
    assert is_module_enabled("rare_mod")

    bus = InMemoryEventBus()
    mgr = RuntimeModuleManager(bus)
    lifecycle = ModuleLifecycleManager(
        event_bus=bus,  # type: ignore[arg-type]
        module_manager=mgr,
        enabled=True,
        disabled_state_path=tmp_path / "disabled.json",
        eval_interval_secs=0.0,
    )
    bus.publish("module.lifecycle.disable", {"module": "rare_mod"})
    bus.join()
    assert not is_module_enabled("rare_mod")

    lifecycle.close()
