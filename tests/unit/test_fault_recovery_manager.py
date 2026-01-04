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

MODULE_PATH = ROOT / "backend" / "execution" / "fault_recovery_manager.py"
spec = util.spec_from_file_location("backend.execution.fault_recovery_manager", MODULE_PATH)
module = util.module_from_spec(spec)
assert spec and spec.loader
sys.modules.setdefault("backend.execution.fault_recovery_manager", module)
spec.loader.exec_module(module)

FaultRecoveryManager = module.FaultRecoveryManager


class DummyModuleManager:
    def __init__(self) -> None:
        self.unloaded: List[str] = []
        self.loaded: List[str] = []

    def unload(self, name: str) -> None:
        self.unloaded.append(name)

    def load(self, name: str) -> None:
        self.loaded.append(name)


def test_fault_recovery_reloads_module_on_failure_burst() -> None:
    bus = InMemoryEventBus()
    attempted: List[Dict[str, Any]] = []
    reloaded: List[Dict[str, Any]] = []
    bus.subscribe("fault_recovery.module_reload_attempted", lambda e: attempted.append(e))
    bus.subscribe("fault_recovery.module_reloaded", lambda e: reloaded.append(e))

    modules = DummyModuleManager()
    recovery = FaultRecoveryManager(
        event_bus=bus,  # type: ignore[arg-type]
        module_manager=modules,
        enabled=True,
        window_secs=60.0,
        max_failures=2,
        cooldown_secs=0.0,
    )
    try:
        bus.publish(
            "task_manager.task_completed",
            {"status": "failed", "time": 1.0, "metadata": {"module": "demo_mod"}, "error": "boom"},
        )
        bus.publish(
            "task_manager.task_completed",
            {"status": "failed", "time": 2.0, "metadata": {"module": "demo_mod"}, "error": "boom"},
        )
        bus.join()

        assert modules.unloaded == ["demo_mod"]
        assert modules.loaded == ["demo_mod"]
        assert attempted and attempted[0]["module"] == "demo_mod"
        assert reloaded and reloaded[-1]["status"] == "ok"
    finally:
        recovery.close()


def test_fault_recovery_ignores_successful_tasks() -> None:
    bus = InMemoryEventBus()
    modules = DummyModuleManager()
    recovery = FaultRecoveryManager(
        event_bus=bus,  # type: ignore[arg-type]
        module_manager=modules,
        enabled=True,
        window_secs=60.0,
        max_failures=1,
        cooldown_secs=0.0,
    )
    try:
        bus.publish(
            "task_manager.task_completed",
            {"status": "success", "time": 1.0, "metadata": {"module": "demo_mod"}},
        )
        bus.join()
        assert modules.loaded == []
        assert modules.unloaded == []
    finally:
        recovery.close()

