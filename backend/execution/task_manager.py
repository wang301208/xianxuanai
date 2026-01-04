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

try:  # Optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

def _torch_gpu_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _gpu_memory_pressures() -> list[float]:
    """Return per-GPU memory pressure ratios (used/total) if available."""

    try:
        import torch  # type: ignore
    except Exception:
        return []
    if not bool(getattr(getattr(torch, "cuda", None), "is_available", lambda: False)()):
        return []
    try:
        count = int(getattr(torch.cuda, "device_count", lambda: 0)() or 0)
    except Exception:
        count = 0
    pressures: list[float] = []
    for idx in range(max(0, count)):
        free = None
        total = None
        try:
            free, total = torch.cuda.mem_get_info(idx)
        except Exception:
            try:
                torch.cuda.set_device(idx)
                free, total = torch.cuda.mem_get_info()
            except Exception:
                continue
        try:
            free_f = float(free or 0.0)
            total_f = float(total or 0.0)
        except Exception:
            continue
        if total_f <= 0:
            continue
        used_ratio = (total_f - free_f) / total_f
        pressures.append(max(0.0, min(1.0, float(used_ratio))))
    return pressures


def _gpu_overloaded(threshold: float) -> bool:
    """Return True when all visible GPUs exceed the provided pressure threshold."""

    pressures = _gpu_memory_pressures()
    if not pressures:
        return False
    try:
        limit = max(0.0, min(1.0, float(threshold)))
    except Exception:
        limit = 0.9
    return all(p >= limit for p in pressures)


class TaskPriority(enum.IntEnum):
    """Enumerate priority bands for managed work."""

    LOW = 10
    NORMAL = 20
    HIGH = 30
    CRITICAL = 40


class AdjustableSemaphore:
    """Semaphore-like concurrency limiter with a mutable upper bound."""

    def __init__(self, limit: int) -> None:
        self._cond = threading.Condition()
        self._limit = max(1, int(limit))
        self._in_use = 0

    @property
    def limit(self) -> int:
        with self._cond:
            return int(self._limit)

    @property
    def in_use(self) -> int:
        with self._cond:
            return int(self._in_use)

    def set_limit(self, limit: int) -> None:
        with self._cond:
            self._limit = max(1, int(limit))
            self._cond.notify_all()

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        if not blocking and timeout is not None:
            raise ValueError("can't specify a timeout for a non-blocking acquire")

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._cond:
            while self._in_use >= self._limit:
                if not blocking:
                    return False
                if deadline is None:
                    self._cond.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
            self._in_use += 1
            return True

    def release(self) -> None:
        with self._cond:
            if self._in_use <= 0:
                self._in_use = 0
                return
            self._in_use -= 1
            self._cond.notify()


@dataclass
class DeviceConfig:
    """Execution resource configuration."""

    executor: concurrent.futures.Executor
    max_workers: int
    semaphore: AdjustableSemaphore = field(init=False)

    def __post_init__(self) -> None:
        self.semaphore = AdjustableSemaphore(self.max_workers)


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
        self._process = None
        if psutil is not None:
            try:
                self._process = psutil.Process(os.getpid())
            except Exception:
                self._process = None
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
        old_executor: concurrent.futures.Executor | None = None
        with self._lock:
            existing = self._devices.get(name)
            if existing is None:
                self._devices[name] = config
            else:
                old_executor = existing.executor
                old_max_workers = int(existing.max_workers)
                old_limit = int(existing.semaphore.limit)
                existing.executor = executor
                existing.max_workers = worker_count
                new_limit = worker_count if old_limit == old_max_workers else min(old_limit, worker_count)
                existing.semaphore.set_limit(new_limit)

        if old_executor is not None:
            try:
                old_executor.shutdown(wait=False)
            except Exception:  # pragma: no cover - best effort cleanup
                pass

    def set_device_concurrency_limit(
        self,
        name: str,
        limit: int,
        *,
        reason: str | None = None,
        source: str | None = None,
    ) -> int:
        """Adjust the per-device concurrency gate without rebuilding executors."""

        name = name or "cpu"
        try:
            desired = int(limit)
        except Exception as exc:
            raise TypeError("limit must be an int") from exc

        self._ensure_device(name)
        changed = False
        previous = 0
        max_workers = 0
        active = 0
        with self._lock:
            cfg = self._devices.get(name)
            if cfg is None:
                return 0
            previous = int(cfg.semaphore.limit)
            max_workers = int(cfg.max_workers)
            desired = max(1, min(desired, max_workers))
            if desired != previous:
                cfg.semaphore.set_limit(desired)
                changed = True
            active = int(cfg.semaphore.in_use)

        if changed and self._event_bus:
            payload: Dict[str, Any] = {
                "time": time.time(),
                "device": name,
                "max_workers": max_workers,
                "previous_limit": previous,
                "concurrency_limit": desired,
                "active": active,
            }
            if reason:
                payload["reason"] = str(reason)
            if source:
                payload["source"] = str(source)
            try:
                self._event_bus.publish("task_manager.device_concurrency_updated", payload)
            except Exception:  # pragma: no cover - optional telemetry
                logger.debug("Failed to publish concurrency update", exc_info=True)

        return int(desired)

    def device_concurrency_limit(self, name: str) -> int:
        """Return the current concurrency limit for *name* (best-effort)."""

        name = name or "cpu"
        with self._lock:
            cfg = self._devices.get(name)
            if cfg is None:
                return 0
            return int(cfg.semaphore.limit)

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
        meta_dict = dict(metadata or {})
        resolved_device = self._resolve_device(
            device,
            priority=priority_value,
            metadata=meta_dict,
        )

        task = ScheduledTask(
            task_id=task_id,
            name=name or getattr(func, "__name__", task_id),
            category=category,
            priority=priority_value,
            deadline=deadline_value,
            created_at=scheduled_at,
            device=resolved_device,
            func=func,
            args=tuple(args),
            kwargs=dict(kwargs),
            future=future,
            metadata=meta_dict,
        )

        self._ensure_device(resolved_device)

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
                    "device": resolved_device,
                    "metadata": task.metadata,
                },
            )

        return TaskHandle(task, self)

    def _resolve_device(self, device: str, *, priority: int, metadata: Dict[str, Any]) -> str:
        requested = (device or "cpu").strip().lower()
        if requested not in {"auto", "any", "best"}:
            return requested or "cpu"

        gpu_capable = bool(metadata.get("gpu_capable") or metadata.get("prefer_gpu"))
        gpu_required = bool(metadata.get("gpu_required"))
        allow_cpu_fallback = bool(metadata.get("allow_cpu_fallback", True))

        threshold_raw = metadata.get("gpu_overload_threshold", os.getenv("TASK_MANAGER_GPU_OVERLOAD_THRESHOLD", "0.9"))
        try:
            gpu_overload_threshold = float(threshold_raw)
        except Exception:
            gpu_overload_threshold = 0.9

        if not gpu_capable and not gpu_required:
            return "cpu"

        if not _torch_gpu_available():
            if gpu_required and not allow_cpu_fallback:
                raise RuntimeError("Task requires GPU but no CUDA device is available.")
            return "cpu"

        self._ensure_gpu_device()
        gpu_overloaded = _gpu_overloaded(gpu_overload_threshold)

        if gpu_required:
            return "gpu"
        if gpu_overloaded and allow_cpu_fallback:
            return "cpu"
        return "gpu"

    def _ensure_gpu_device(self) -> None:
        with self._lock:
            if "gpu" in self._devices:
                return
        if not _torch_gpu_available():
            return
        workers = 1
        try:
            import torch  # type: ignore

            workers = int(torch.cuda.device_count() or 1)
        except Exception:
            workers = 1
        self.configure_device("gpu", max_workers=max(1, workers))

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
        scan_limit = int(os.getenv("TASK_MANAGER_DISPATCH_SCAN_LIMIT", "32") or 32)
        scan_limit = max(1, min(scan_limit, 512))
        idle_sleep = float(os.getenv("TASK_MANAGER_DISPATCH_IDLE_SLEEP", "0.01") or 0.01)
        idle_sleep = max(0.0, min(idle_sleep, 0.5))

        while not self._stop_event.is_set():
            deferred: list[ScheduledTask] = []
            dispatched = False

            for scan in range(scan_limit):
                try:
                    timeout = 0.5 if scan == 0 else 0.0
                    _, _, _, _, task = self._queue.get(timeout=timeout)
                except queue.Empty:
                    break

                if task is self._SENTINEL:
                    self._queue.task_done()
                    return

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
                acquired = device_cfg.semaphore.acquire(blocking=False)
                if not acquired:
                    deferred.append(task)
                    self._queue.task_done()
                    continue

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
                dispatched = True
                break

            if deferred:
                with self._lock:
                    for task in deferred:
                        if task.cancelled or task.task_id not in self._tasks:
                            continue
                        entry = (
                            -int(task.priority),
                            float(task.deadline),
                            float(task.created_at),
                            next(self._order_counter),
                            task,
                        )
                        self._queue.put(entry)

            if not dispatched and idle_sleep > 0:
                time.sleep(idle_sleep)

    def _run_task(self, device_cfg: DeviceConfig, task: ScheduledTask) -> None:
        status = "completed"
        started_at = time.time()
        started_perf = time.perf_counter()
        started_cpu = _thread_cpu_time()
        started_rss = _process_rss_bytes(self._process)
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
            ended_at = time.time()
            ended_perf = time.perf_counter()
            ended_cpu = _thread_cpu_time()
            ended_rss = _process_rss_bytes(self._process)

            duration_s = max(0.0, ended_perf - started_perf)
            cpu_time_s = max(0.0, ended_cpu - started_cpu)
            cpu_percent = 0.0
            if duration_s > 0.0:
                cpu_percent = max(0.0, (cpu_time_s / duration_s) * 100.0)

            rss_delta_bytes = None
            if started_rss is not None and ended_rss is not None:
                rss_delta_bytes = int(ended_rss - started_rss)

            device_cfg.semaphore.release()
            with self._lock:
                self._active = max(0, self._active - 1)
                self._tasks.pop(task.task_id, None)
                depth = self._queue.qsize() + self._active
            self._notify_depth(depth)
            if self._event_bus:
                payload = {
                    "time": ended_at,
                    "started_at": started_at,
                    "duration_s": duration_s,
                    "task_id": task.task_id,
                    "name": task.name,
                    "category": task.category,
                    "status": status,
                    "device": task.device,
                    "metadata": task.metadata,
                    "runtime": {
                        "cpu_time_s": cpu_time_s,
                        "cpu_percent": cpu_percent,
                        "rss_start_bytes": started_rss,
                        "rss_end_bytes": ended_rss,
                        "rss_delta_bytes": rss_delta_bytes,
                    },
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


def _thread_cpu_time() -> float:
    """Return CPU time for the current worker thread."""

    thread_time = getattr(time, "thread_time", None)
    if callable(thread_time):
        try:
            return float(thread_time())
        except Exception:
            pass
    try:
        return float(time.process_time())
    except Exception:
        return 0.0


def _process_rss_bytes(process: Any | None) -> int | None:
    """Best-effort RSS measurement for the current process."""

    if process is None:
        return None
    memory_info = getattr(process, "memory_info", None)
    if not callable(memory_info):
        return None
    try:
        info = memory_info()
        rss = getattr(info, "rss", None)
        if rss is None:
            return None
        return int(rss)
    except Exception:
        return None
