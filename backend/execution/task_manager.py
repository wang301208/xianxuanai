"""Priority-based asynchronous task management for cognitive workloads."""
from __future__ import annotations

import asyncio
import concurrent.futures
import enum
import inspect
import logging
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Callable, Dict, Optional, Tuple

try:  # The event bus is optional in some deployments
    from events import EventBus  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    EventBus = None  # type: ignore

logger = logging.getLogger(__name__)

CallableTask = Callable[..., Any]


class TaskPriority(enum.IntEnum):
    """Enumerate priority bands for managed work."""

    LOW = 10
    NORMAL = 20
    HIGH = 30
    CRITICAL = 40


@dataclass
class DeviceConfig:
    """Execution resource configuration."""

    executor: concurrent.futures.Executor
    max_workers: int
    semaphore: threading.Semaphore = field(init=False)

    def __post_init__(self) -> None:
        self.semaphore = threading.Semaphore(self.max_workers)


@dataclass
class ScheduledTask:
    """Metadata tracked for each queued task."""

    task_id: str
    name: str
    category: str
    priority: int
    deadline: float
    created_at: float
    device: str
    func: CallableTask
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    future: concurrent.futures.Future
    metadata: Dict[str, Any]
    cancelled: bool = False


class TaskHandle:
    """Wrapper returned to callers for monitoring and cancellation."""

    def __init__(self, task: ScheduledTask, manager: "TaskManager") -> None:
        self._task = task
        self._future = task.future
        self._manager = manager

    @property
    def task_id(self) -> str:
        return self._task.task_id

    @property
    def name(self) -> str:
        return self._task.name

    @property
    def category(self) -> str:
        return self._task.category

    @property
    def priority(self) -> int:
        return self._task.priority

    @property
    def deadline(self) -> float:
        return self._task.deadline

    @property
    def device(self) -> str:
        return self._task.device

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._task.metadata)

    def cancel(self) -> bool:
        """Attempt to cancel the task if it has not started yet."""

        return self._manager._cancel_task(self._task.task_id)

    def done(self) -> bool:
        return self._future.done()

    def result(self, timeout: Optional[float] = None) -> Any:
        return self._future.result(timeout)

    def exception(self, timeout: Optional[float] = None) -> Optional[BaseException]:
        return self._future.exception(timeout)

    def add_done_callback(self, callback: Callable[["TaskHandle"], None]) -> None:
        """Invoke *callback* when the underlying future completes."""

        def _wrapped(_: concurrent.futures.Future) -> None:
            callback(self)

        self._future.add_done_callback(_wrapped)


class TaskManager:
    """Manage heterogeneous workloads with prioritised scheduling."""

    _SENTINEL = object()

    def __init__(
        self,
        *,
        event_bus: Optional[EventBus] = None,
        queue_callback: Optional[Callable[[int], None]] = None,
        default_cpu_workers: Optional[int] = None,
        resource_id: Optional[str] = None,
    ) -> None:
        self._event_bus = event_bus
        self._queue_callback = queue_callback
        self._resource_id = resource_id
        self._queue: "queue.PriorityQueue[tuple[Any, Any, Any, int, Any]]" = queue.PriorityQueue()
        self._devices: Dict[str, DeviceConfig] = {}
        self._tasks: Dict[str, ScheduledTask] = {}
        self._lock = threading.RLock()
        self._order_counter = count()
        self._active = 0
        self._stop_event = threading.Event()
        self._dispatcher_thread = threading.Thread(target=self._dispatch_loop, daemon=True, name="TaskManager")
        self._default_workers = default_cpu_workers or max(2, (os.cpu_count() or 2) // 2)
        self.configure_device("cpu", max_workers=self._default_workers)
        self.start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if not self._dispatcher_thread.is_alive():
            self._dispatcher_thread = threading.Thread(target=self._dispatch_loop, daemon=True, name="TaskManager")
            self._dispatcher_thread.start()

    def shutdown(self, wait: bool = True) -> None:
        """Stop dispatching and release all executors."""

        self._stop_event.set()
        # Wake the dispatcher if idle
        self._queue.put((0, float("inf"), 0.0, next(self._order_counter), self._SENTINEL))
        if wait and self._dispatcher_thread.is_alive():
            self._dispatcher_thread.join()

        with self._lock:
            tasks = list(self._tasks.values())
            self._tasks.clear()
        for task in tasks:
            if not task.future.done():
                task.future.cancel()

        devices = []
        with self._lock:
            devices = list(self._devices.values())
            self._devices.clear()
        for device in devices:
            try:
                device.executor.shutdown(wait=wait)
            except Exception:  # pragma: no cover - best effort cleanup
                pass

    close = shutdown

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------
    def configure_device(
        self,
        name: str,
        *,
        max_workers: Optional[int] = None,
        executor: Optional[concurrent.futures.Executor] = None,
    ) -> None:
        """Register or update an execution device."""

        name = name or "cpu"
        if executor is None:
            worker_count = max(1, max_workers or 1)
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix=f"taskmgr-{name}",
            )
        else:
            worker_count = max_workers or getattr(executor, "_max_workers", 1) or 1

        config = DeviceConfig(executor=executor, max_workers=worker_count)
        with self._lock:
            existing = self._devices.get(name)
            self._devices[name] = config
        if existing:
            try:
                existing.executor.shutdown(wait=False)
            except Exception:  # pragma: no cover - best effort cleanup
                pass

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------
    def submit(
        self,
        func: CallableTask,
        *args: Any,
        priority: int | TaskPriority = TaskPriority.NORMAL,
        deadline: Optional[float] = None,
        category: str = "general",
        name: Optional[str] = None,
        device: str = "cpu",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TaskHandle:
        """Schedule *func* for execution with the provided metadata."""

        if not callable(func):
            raise TypeError("func must be callable")

        priority_value = int(priority)
        if deadline is None:
            deadline_value = float("inf")
        else:
            deadline_value = float(deadline)

        scheduled_at = time.time()
        task_id = str(uuid.uuid4())
        future: concurrent.futures.Future = concurrent.futures.Future()

        task = ScheduledTask(
            task_id=task_id,
            name=name or getattr(func, "__name__", task_id),
            category=category,
            priority=priority_value,
            deadline=deadline_value,
            created_at=scheduled_at,
            device=device,
            func=func,
            args=tuple(args),
            kwargs=dict(kwargs),
            future=future,
            metadata=dict(metadata or {}),
        )

        self._ensure_device(device)

        with self._lock:
            self._tasks[task_id] = task
            entry = (-priority_value, deadline_value, scheduled_at, next(self._order_counter), task)
            self._queue.put(entry)
            depth = self._queue.qsize() + self._active
        self._notify_depth(depth)

        if self._event_bus:
            self._event_bus.publish(
                "task_manager.task_scheduled",
                {
                    "task_id": task_id,
                    "name": task.name,
                    "category": category,
                    "priority": priority_value,
                    "deadline": deadline_value,
                    "device": device,
                    "metadata": task.metadata,
                },
            )

        return TaskHandle(task, self)

    def queue_depth(self) -> int:
        with self._lock:
            return self._queue.qsize() + self._active

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_device(self, name: str) -> None:
        with self._lock:
            if name in self._devices:
                return
        self.configure_device(name, max_workers=1)

    def _notify_depth(self, depth: int) -> None:
        if self._queue_callback:
            try:
                self._queue_callback(depth)
            except Exception:  # pragma: no cover - observer failures should not crash
                logger.debug("Queue observer failed", exc_info=True)

        if self._resource_id:
            try:
                from modules.environment import report_resource_signal

                report_resource_signal(
                    self._resource_id,
                    {"queue_depth": float(depth)},
                    metadata={"active_workers": self._active},
                    event_bus=self._event_bus,
                )
            except Exception:  # pragma: no cover - optional telemetry
                logger.debug("Resource signal emission failed", exc_info=True)

    def _dispatch_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                _, _, _, _, task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if task is self._SENTINEL:
                self._queue.task_done()
                break

            if not isinstance(task, ScheduledTask):
                self._queue.task_done()
                continue

            if task.cancelled:
                self._queue.task_done()
                if not task.future.done():
                    task.future.cancel()
                with self._lock:
                    self._tasks.pop(task.task_id, None)
                    depth = self._queue.qsize() + self._active
                self._notify_depth(depth)
                if self._event_bus:
                    self._event_bus.publish(
                        "task_manager.task_cancelled",
                        {"task_id": task.task_id, "name": task.name, "category": task.category},
                    )
                continue

            device = task.device
            with self._lock:
                device_cfg = self._devices.get(device)
            if device_cfg is None:
                self._ensure_device(device)
                with self._lock:
                    device_cfg = self._devices[device]

            assert device_cfg is not None
            device_cfg.semaphore.acquire()
            with self._lock:
                self._active += 1
                depth = self._queue.qsize() + self._active
            self._notify_depth(depth)
            self._queue.task_done()

            if self._event_bus:
                self._event_bus.publish(
                    "task_manager.task_started",
                    {
                        "task_id": task.task_id,
                        "name": task.name,
                        "category": task.category,
                        "device": device,
                        "priority": task.priority,
                    },
                )

            device_cfg.executor.submit(self._run_task, device_cfg, task)

    def _run_task(self, device_cfg: DeviceConfig, task: ScheduledTask) -> None:
        status = "completed"
        try:
            autofix_cfg = None
            if isinstance(task.metadata, dict):
                autofix_cfg = task.metadata.get("autofix")
            if autofix_cfg:
                try:
                    from modules.diagnostics.auto_fixer import execute_with_autofix

                    task_context = {
                        "task_id": task.task_id,
                        "name": task.name,
                        "category": task.category,
                        "device": task.device,
                    }
                    result = execute_with_autofix(task.func, task.args, task.kwargs, autofix_cfg, task_context)
                except Exception:
                    result = task.func(*task.args, **task.kwargs)
            else:
                result = task.func(*task.args, **task.kwargs)
            if inspect.isawaitable(result):
                result = asyncio.run(result)
            if not task.future.cancelled():
                task.future.set_result(result)
        except Exception as exc:
            status = "failed"
            if not task.future.cancelled():
                task.future.set_exception(exc)
            logger.debug("Task %s failed", task.name, exc_info=True)
        finally:
            device_cfg.semaphore.release()
            with self._lock:
                self._active = max(0, self._active - 1)
                self._tasks.pop(task.task_id, None)
                depth = self._queue.qsize() + self._active
            self._notify_depth(depth)
            if self._event_bus:
                payload = {
                    "task_id": task.task_id,
                    "name": task.name,
                    "category": task.category,
                    "status": status,
                    "device": task.device,
                    "metadata": task.metadata,
                }
                if status != "completed" and not task.future.cancelled():
                    exc = task.future.exception(timeout=0)
                    if exc is not None:
                        payload["error"] = repr(exc)
                        try:
                            from modules.diagnostics.auto_fixer import AutoFixFailed, AutoFixer

                            if isinstance(exc, AutoFixFailed):
                                payload["autofix"] = exc.payload()
                            else:
                                autofixer = AutoFixer.from_env()
                                if autofixer is not None:
                                    ctx = {
                                        "task_id": task.task_id,
                                        "name": task.name,
                                        "category": task.category,
                                        "device": task.device,
                                        "metadata": dict(task.metadata or {}),
                                    }
                                    analysis = autofixer.analyze_error(exc, context=ctx)
                                    plan = autofixer.generate_fix_plan(analysis)
                                    if plan is not None:
                                        payload["autofix"] = {
                                            "analysis": analysis.to_dict(),
                                            "fix_history": [plan.to_dict()],
                                        }
                        except Exception:  # pragma: no cover - best effort diagnostics
                            pass
                self._event_bus.publish("task_manager.task_completed", payload)

    def _cancel_task(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.future.done():
                return False
            task.cancelled = True
            depth = self._queue.qsize() + self._active
        self._notify_depth(depth)
        return True


__all__ = ["TaskManager", "TaskPriority", "TaskHandle"]
