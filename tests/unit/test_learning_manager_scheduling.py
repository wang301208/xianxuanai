import asyncio
from dataclasses import dataclass
from importlib import util
from pathlib import Path
import sys
import types
from typing import Any, Callable, Dict, List

ROOT = Path(__file__).resolve().parents[2]
backend_pkg = types.ModuleType("backend")
backend_pkg.__path__ = [str(ROOT / "backend")]
sys.modules.setdefault("backend", backend_pkg)
execution_pkg = types.ModuleType("backend.execution")
execution_pkg.__path__ = [str(ROOT / "backend" / "execution")]
sys.modules.setdefault("backend.execution", execution_pkg)

MODULE_PATH = ROOT / "backend" / "execution" / "learning_manager.py"
spec = util.spec_from_file_location("backend.execution.learning_manager", MODULE_PATH)
learning_manager = util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(learning_manager)
LearningManager = learning_manager.LearningManager


class SyncEventBus:
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[[Dict[str, Any]], Any]]] = {}

    def publish(self, topic: str, event: Dict[str, Any]) -> None:
        for handler in list(self._subscribers.get(topic, [])):
            result = handler(event)
            if asyncio.iscoroutine(result):
                asyncio.run(result)

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], Any]) -> Callable[[], None]:
        self._subscribers.setdefault(topic, []).append(handler)
        return lambda: self.unsubscribe(topic, handler)

    def unsubscribe(self, topic: str, handler: Callable[[Dict[str, Any]], Any]) -> None:
        handlers = self._subscribers.get(topic, [])
        if handler in handlers:
            handlers.remove(handler)
        if not handlers and topic in self._subscribers:
            del self._subscribers[topic]


@dataclass
class DummyHandle:
    task_id: str
    metadata: Dict[str, Any]
    _result: Any = None
    _callbacks: List[Callable[["DummyHandle"], None]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._callbacks is None:
            self._callbacks = []

    def add_done_callback(self, callback: Callable[["DummyHandle"], None]) -> None:
        self._callbacks.append(callback)

    def result(self, timeout: float | None = None) -> Any:
        del timeout
        return self._result

    def finish(self, result: Any) -> None:
        self._result = result
        for cb in list(self._callbacks):
            cb(self)


class DummyTaskManager:
    def __init__(self) -> None:
        self.submissions: List[Dict[str, Any]] = []
        self.handles: List[DummyHandle] = []

    def submit(self, func: Callable[[], Any], *args: Any, **kwargs: Any) -> DummyHandle:
        del args
        handle = DummyHandle(task_id=f"task-{len(self.handles) + 1}", metadata=kwargs.get("metadata") or {})
        self.submissions.append({"func": func, "kwargs": dict(kwargs), "handle": handle})
        self.handles.append(handle)
        return handle


def test_learning_manager_schedules_maintenance_on_tick():
    bus = SyncEventBus()
    tasks = DummyTaskManager()
    manager = LearningManager(
        event_bus=bus,  # type: ignore[arg-type]
        task_manager=tasks,  # type: ignore[arg-type]
        run_learning_cycle=lambda: {"ok": True},
        min_interval=0.0,
        event_min_interval=0.0,
        token_capacity=1.0,
        token_refill_per_sec=0.0,
        cpu_ceiling=99.0,
        mem_ceiling=99.0,
    )

    bus.publish("learning.tick", {"avg_cpu": 0.0, "avg_memory": 0.0, "backlog": 0, "time": 10.0})

    assert len(tasks.submissions) == 1
    metadata = tasks.submissions[0]["kwargs"]["metadata"]
    assert metadata["reason"] == "maintenance"

    # No tokens left, so additional ticks do not schedule.
    bus.publish("learning.tick", {"avg_cpu": 0.0, "avg_memory": 0.0, "backlog": 0, "time": 11.0})
    assert len(tasks.submissions) == 1
    manager.close()


def test_learning_manager_schedules_after_task_completed_request():
    bus = SyncEventBus()
    tasks = DummyTaskManager()
    manager = LearningManager(
        event_bus=bus,  # type: ignore[arg-type]
        task_manager=tasks,  # type: ignore[arg-type]
        run_learning_cycle=lambda: {"ok": True},
        min_interval=1000.0,  # block maintenance scheduling in this test
        event_min_interval=0.0,
        token_capacity=2.0,
        token_refill_per_sec=0.0,
        cpu_ceiling=99.0,
        mem_ceiling=99.0,
    )

    bus.publish("learning.tick", {"avg_cpu": 0.0, "avg_memory": 0.0, "backlog": 0, "time": 10.0})
    assert len(tasks.submissions) == 0

    bus.publish("coordinator.task_completed", {"status": "completed"})
    bus.publish("learning.tick", {"avg_cpu": 0.0, "avg_memory": 0.0, "backlog": 0, "time": 20.0})

    assert len(tasks.submissions) == 1
    metadata = tasks.submissions[0]["kwargs"]["metadata"]
    assert metadata["reason"] == "event"
    assert metadata["reasons"].get("task_completed") == 1
    manager.close()


def test_learning_manager_defers_until_backlog_clears():
    bus = SyncEventBus()
    tasks = DummyTaskManager()
    manager = LearningManager(
        event_bus=bus,  # type: ignore[arg-type]
        task_manager=tasks,  # type: ignore[arg-type]
        run_learning_cycle=lambda: {"ok": True},
        min_interval=1000.0,
        event_min_interval=0.0,
        token_capacity=2.0,
        token_refill_per_sec=0.0,
        cpu_ceiling=99.0,
        mem_ceiling=99.0,
    )

    bus.publish("coordinator.task_completed", {"status": "completed"})
    bus.publish("learning.tick", {"avg_cpu": 0.0, "avg_memory": 0.0, "backlog": 2, "time": 20.0})
    assert len(tasks.submissions) == 0

    bus.publish("learning.tick", {"avg_cpu": 0.0, "avg_memory": 0.0, "backlog": 0, "time": 21.0})
    assert len(tasks.submissions) == 1
    assert tasks.submissions[0]["kwargs"]["metadata"]["reasons"].get("task_completed") == 1
    manager.close()


def test_learning_manager_throttle_limits_tokens():
    bus = SyncEventBus()
    tasks = DummyTaskManager()
    manager = LearningManager(
        event_bus=bus,  # type: ignore[arg-type]
        task_manager=tasks,  # type: ignore[arg-type]
        run_learning_cycle=lambda: {"ok": True},
        min_interval=0.0,
        event_min_interval=0.0,
        token_capacity=10.0,
        token_refill_per_sec=0.0,
        cpu_ceiling=99.0,
        mem_ceiling=99.0,
    )

    manager.throttle(max_tokens=0.5)
    assert manager.status()["tokens"] <= 0.5
    manager.close()
