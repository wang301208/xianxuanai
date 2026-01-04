from typing import Any, Dict, List

from modules.events import InMemoryEventBus

from backend.execution.scheduler_control_manager import SchedulerControlManager
from backend.execution.task_manager import TaskManager
from backend.capability import register_module
from backend.capability.runtime_loader import RuntimeModuleManager


def test_scheduler_control_throttle_updates_task_manager_limits() -> None:
    bus = InMemoryEventBus()
    statuses: List[Dict[str, Any]] = []
    bus.subscribe("scheduler.status", lambda e: statuses.append(e))

    task_manager = TaskManager(event_bus=bus, default_cpu_workers=4)
    control = SchedulerControlManager(event_bus=bus, task_manager=task_manager, enabled=True)
    try:
        bus.publish(
            "scheduler.control",
            {"action": "throttle", "device": "cpu", "concurrency": 1, "reason": "test"},
        )
        bus.join()

        assert task_manager.device_concurrency_limit("cpu") == 1
        assert statuses
        assert any(item.get("trigger") == "throttle" for item in statuses)
    finally:
        control.close()
        task_manager.shutdown()


def test_scheduler_control_module_update_config_calls_module() -> None:
    bus = InMemoryEventBus()

    name = "dummy_scheduler_control_update_config"
    calls: list[dict] = []

    class DummyModule:
        def update_config(self, runtime_config=None, *, overrides=None):  # pragma: no cover - test hook
            calls.append({"runtime_config": runtime_config, "overrides": overrides})

    register_module(name, DummyModule)

    module_manager = RuntimeModuleManager(bus)
    control = SchedulerControlManager(event_bus=bus, module_manager=module_manager, enabled=True)
    try:
        bus.publish(
            "scheduler.control",
            {"action": "module.update_config", "module": name, "overrides": {"x": 1}, "load": False},
        )
        bus.join()
        assert name not in module_manager.loaded_modules()
        assert not calls

        bus.publish(
            "scheduler.control",
            {"action": "module.update_config", "module": name, "overrides": {"x": 2}, "load": True},
        )
        bus.join()
        assert name in module_manager.loaded_modules()
        assert calls and calls[-1]["overrides"]["x"] == 2
    finally:
        control.close()
        if name in module_manager.loaded_modules():
            module_manager.unload(name)
