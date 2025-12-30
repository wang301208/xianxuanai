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
