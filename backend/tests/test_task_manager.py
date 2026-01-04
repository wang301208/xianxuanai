import asyncio
import importlib.util
import os
import sys
import threading
import time
from pathlib import Path
from typing import List

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

BACKEND_EXECUTION = Path(ROOT, "backend", "execution")


def _load(name: str):
    module_path = BACKEND_EXECUTION / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"backend.execution.{name}", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


task_manager_module = _load("task_manager")
TaskManager = task_manager_module.TaskManager
TaskPriority = task_manager_module.TaskPriority


@pytest.fixture()
def manager() -> TaskManager:
    mgr = TaskManager()
    mgr.configure_device("cpu", max_workers=1)
    yield mgr
    mgr.shutdown()


def test_priority_dispatch_order(manager: TaskManager) -> None:
    order: List[str] = []

    def record(name: str) -> str:
        order.append(name)
        return name

    low = manager.submit(record, "low", priority=TaskPriority.LOW, name="low")
    high = manager.submit(record, "high", priority=TaskPriority.CRITICAL, name="high")

    assert high.result(timeout=1.0) == "high"
    assert low.result(timeout=1.0) == "low"
    assert order == ["high", "low"]


def test_deadline_breaks_priority_ties(manager: TaskManager) -> None:
    order: List[str] = []
    now = time.time()

    def record(name: str) -> str:
        order.append(name)
        return name

    late = manager.submit(
        record,
        "late",
        priority=TaskPriority.NORMAL,
        deadline=now + 10,
        name="late",
    )
    early = manager.submit(
        record,
        "early",
        priority=TaskPriority.NORMAL,
        deadline=now + 1,
        name="early",
    )

    assert early.result(timeout=1.0) == "early"
    assert late.result(timeout=1.0) == "late"
    assert order == ["early", "late"]


def test_device_concurrency_respected(manager: TaskManager) -> None:
    manager.configure_device("gpu", max_workers=1)
    start_order: List[str] = []
    gate = threading.Event()

    def long_task() -> str:
        start_order.append("first")
        gate.wait(timeout=1.0)
        return "long"

    def short_task() -> str:
        start_order.append("second")
        return "short"

    first = manager.submit(long_task, priority=TaskPriority.HIGH, device="gpu")
    time.sleep(0.05)
    second = manager.submit(short_task, priority=TaskPriority.HIGH, device="gpu")

    time.sleep(0.1)
    assert start_order == ["first"]
    gate.set()

    assert first.result(timeout=1.0) == "long"
    assert second.result(timeout=1.0) == "short"
    assert start_order == ["first", "second"]


def test_dynamic_device_concurrency_limit_blocks_new_starts() -> None:
    mgr = TaskManager()
    mgr.configure_device("cpu", max_workers=2)
    started: list[str] = []
    gate1 = threading.Event()
    gate2 = threading.Event()

    def long_task(label: str, gate: threading.Event) -> str:
        started.append(label)
        gate.wait(timeout=1.0)
        return label

    first = mgr.submit(long_task, "t1", gate1, priority=TaskPriority.HIGH, device="cpu")
    second = mgr.submit(long_task, "t2", gate2, priority=TaskPriority.HIGH, device="cpu")

    for _ in range(50):
        if len(started) >= 2:
            break
        time.sleep(0.02)
    assert set(started) == {"t1", "t2"}

    mgr.set_device_concurrency_limit("cpu", 1, reason="high_cpu", source="test")

    def third_task() -> str:
        started.append("t3")
        return "t3"

    third = mgr.submit(third_task, priority=TaskPriority.HIGH, device="cpu")
    time.sleep(0.1)
    assert "t3" not in started

    gate1.set()
    assert first.result(timeout=1.0) in {"t1", "t2"}
    time.sleep(0.1)
    assert "t3" not in started

    gate2.set()
    assert second.result(timeout=1.0) in {"t1", "t2"}
    assert third.result(timeout=1.0) == "t3"
    mgr.shutdown()


def test_auto_device_routes_gpu_capable_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = TaskManager()
    mgr.configure_device("cpu", max_workers=1)
    mgr.configure_device("gpu", max_workers=1)

    monkeypatch.setattr(task_manager_module, "_torch_gpu_available", lambda: True)
    monkeypatch.setattr(task_manager_module, "_gpu_overloaded", lambda _threshold: False)

    def identify() -> str:
        return threading.current_thread().name

    handle = mgr.submit(identify, priority=TaskPriority.HIGH, device="auto", metadata={"gpu_capable": True})
    thread_name = handle.result(timeout=1.0)
    assert thread_name.startswith("taskmgr-gpu")

    monkeypatch.setattr(task_manager_module, "_gpu_overloaded", lambda _threshold: True)
    handle2 = mgr.submit(identify, priority=TaskPriority.HIGH, device="auto", metadata={"gpu_capable": True})
    thread_name2 = handle2.result(timeout=1.0)
    assert thread_name2.startswith("taskmgr-cpu")

    mgr.shutdown()


def test_dispatcher_does_not_head_of_line_block_across_devices() -> None:
    mgr = TaskManager()
    mgr.configure_device("cpu", max_workers=1)
    mgr.configure_device("gpu", max_workers=1)

    started: list[str] = []
    gate = threading.Event()

    def long_gpu() -> str:
        started.append("gpu1")
        gate.wait(timeout=1.0)
        return "gpu1"

    def queued_gpu() -> str:
        started.append("gpu2")
        return "gpu2"

    def cpu_task() -> str:
        started.append("cpu")
        return "cpu"

    first = mgr.submit(long_gpu, priority=TaskPriority.HIGH, device="gpu")
    for _ in range(50):
        if "gpu1" in started:
            break
        time.sleep(0.02)
    assert "gpu1" in started

    second = mgr.submit(queued_gpu, priority=TaskPriority.HIGH, device="gpu")
    third = mgr.submit(cpu_task, priority=TaskPriority.HIGH, device="cpu")

    time.sleep(0.1)
    assert "cpu" in started
    assert "gpu2" not in started

    gate.set()

    assert first.result(timeout=1.0) == "gpu1"
    assert third.result(timeout=1.0) == "cpu"
    assert second.result(timeout=1.0) == "gpu2"
    mgr.shutdown()


def test_async_callable_supported(manager: TaskManager) -> None:
    async def coro(value: int) -> int:
        await asyncio.sleep(0)
        return value * 2

    handle = manager.submit(coro, 21, priority=TaskPriority.NORMAL, name="async")
    assert handle.result(timeout=1.0) == 42


def test_queue_observer_reports_depth() -> None:
    depths: List[int] = []
    mgr = TaskManager(queue_callback=lambda depth: depths.append(depth))
    mgr.configure_device("cpu", max_workers=1)
    handle = mgr.submit(lambda: "done", priority=TaskPriority.NORMAL, name="observer")
    assert handle.result(timeout=1.0) == "done"
    time.sleep(0.05)
    mgr.shutdown()
    assert depths and depths[-1] == 0


def test_task_completed_event_includes_runtime_metrics() -> None:
    class CaptureBus:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict]] = []
            self._lock = threading.Lock()

        def publish(self, topic: str, event: dict) -> None:
            with self._lock:
                self.events.append((topic, dict(event)))

    bus = CaptureBus()
    mgr = TaskManager(event_bus=bus)
    mgr.configure_device("cpu", max_workers=1)

    handle = mgr.submit(lambda: "ok", priority=TaskPriority.NORMAL, name="metrics", category="unit")
    assert handle.result(timeout=1.0) == "ok"

    mgr.shutdown()
    completed = [
        event for topic, event in bus.events
        if topic == "task_manager.task_completed" and event.get("task_id") == handle.task_id
    ]
    assert completed
    payload = completed[-1]
    runtime = payload.get("runtime") or {}
    assert payload.get("duration_s") is not None
    assert float(runtime.get("cpu_time_s", 0.0)) >= 0.0
    assert float(runtime.get("cpu_percent", 0.0)) >= 0.0
