
from __future__ import annotations

"""Pluggable task adapters for local, Ray or Dask execution."""

import concurrent.futures
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

try:  # Optional event bus dependency
    from modules.events import EventBus, publish
except Exception:  # pragma: no cover - optional dependency
    EventBus = None  # type: ignore
    publish = None  # type: ignore

try:  # Optional Ray dependency
    import ray  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ray = None  # type: ignore

try:  # Optional Dask dependency
    from dask.distributed import Client as DaskClient  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    DaskClient = None  # type: ignore

from modules.environment.registry import get_hardware_registry, report_resource_signal


TaskCallable = Callable[..., Any]


class TaskFuture(ABC):
    """Unified Future API used by task adapters."""

    @abstractmethod
    def result(self, timeout: Optional[float] = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def exception(self, timeout: Optional[float] = None) -> Optional[BaseException]:
        raise NotImplementedError

    @abstractmethod
    def done(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def cancel(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def add_done_callback(self, callback: Callable[[TaskFuture], None]) -> None:
        raise NotImplementedError


class _LocalTaskFuture(TaskFuture):
    def __init__(self, future: concurrent.futures.Future):
        self._future = future

    def result(self, timeout: Optional[float] = None) -> Any:
        return self._future.result(timeout)

    def exception(self, timeout: Optional[float] = None) -> Optional[BaseException]:
        return self._future.exception(timeout)

    def done(self) -> bool:
        return self._future.done()

    def cancel(self) -> bool:
        return self._future.cancel()

    def add_done_callback(self, callback: Callable[[TaskFuture], None]) -> None:
        def _wrapped(_: concurrent.futures.Future) -> None:
            callback(self)

        self._future.add_done_callback(_wrapped)


class _RayTaskFuture(TaskFuture):
    def __init__(self, object_ref):
        self._ref = object_ref

    def result(self, timeout: Optional[float] = None) -> Any:
        if ray is None:  # pragma: no cover - defensive
            raise RuntimeError("Ray not available")
        try:
            return ray.get(self._ref, timeout=timeout)
        except ray.exceptions.GetTimeoutError as exc:
            raise TimeoutError(str(exc)) from exc

    def exception(self, timeout: Optional[float] = None) -> Optional[BaseException]:
        try:
            self.result(timeout)
            return None
        except Exception as exc:
            return exc

    def done(self) -> bool:
        if ray is None:  # pragma: no cover
            return False
        ready, _ = ray.wait([self._ref], timeout=0)
        return bool(ready)

    def cancel(self) -> bool:
        # Best-effort cancellation; ray doesn't guarantee cancellation.
        if ray is None:  # pragma: no cover
            return False
        try:
            ray.cancel(self._ref, force=True)
            return True
        except Exception:
            return False

    def add_done_callback(self, callback: Callable[[TaskFuture], None]) -> None:
        if ray is None:  # pragma: no cover
            return

        def _wait_and_callback() -> None:
            ray.wait([self._ref], timeout=None)
            callback(self)

        threading.Thread(target=_wait_and_callback, daemon=True).start()


class _DaskTaskFuture(TaskFuture):
    def __init__(self, future):
        self._future = future

    def result(self, timeout: Optional[float] = None) -> Any:
        return self._future.result(timeout=timeout)

    def exception(self, timeout: Optional[float] = None) -> Optional[BaseException]:
        try:
            self.result(timeout)
            return None
        except Exception as exc:
            return exc

    def done(self) -> bool:
        return self._future.done()

    def cancel(self) -> bool:
        try:
            self._future.cancel()
            return True
        except Exception:
            return False

    def add_done_callback(self, callback: Callable[[TaskFuture], None]) -> None:
        self._future.add_done_callback(lambda _: callback(self))


class TaskAdapter(ABC):
    """Base class for pluggable task execution backends."""

    def __init__(
        self,
        *,
        worker_id: Optional[str] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self.worker_id = worker_id or f"task-{uuid.uuid4().hex[:8]}"
        self.event_bus = event_bus
        self._lock = threading.Lock()
        self._pending = 0
        self._completed = 0
        self._failed = 0
        self._autofix_snapshots: Dict[str, Dict[str, Any]] = {}
        self._register_worker()

    # ------------------------------------------------------------------
    def submit(
        self,
        func: TaskCallable,
        *args: Any,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TaskFuture:
        metadata = dict(metadata or {})
        task_id = metadata.get("task_id") or uuid.uuid4().hex
        name = metadata.get("name", getattr(func, "__name__", "task"))
        category = metadata.get("category", "generic")

        with self._lock:
            self._pending += 1

        self._publish_event(
            "task.dispatch",
            {
                "task_id": task_id,
                "worker_id": self.worker_id,
                "name": name,
                "category": category,
                "metadata": metadata,
            },
        )

        autofix_cfg = metadata.get("autofix")
        autofix_strategy = "in_task"
        if isinstance(autofix_cfg, dict):
            autofix_strategy = str(autofix_cfg.get("strategy") or autofix_cfg.get("mode") or "in_task").lower()
        if autofix_cfg and "autofix_attempt" not in metadata:
            metadata["autofix_attempt"] = 0

        if autofix_cfg:
            with self._lock:
                self._autofix_snapshots[task_id] = {
                    "func": func,
                    "args": args,
                    "kwargs": kwargs,
                    "name": name,
                    "category": category,
                    "metadata": dict(metadata),
                }

        if autofix_cfg and autofix_strategy != "resubmit":
            try:
                from modules.diagnostics.auto_fixer import execute_with_autofix

                task_context = {
                    "task_id": task_id,
                    "worker_id": self.worker_id,
                    "name": name,
                    "category": category,
                }
                future = self._submit_impl(
                    execute_with_autofix,
                    (func, args, kwargs, autofix_cfg, task_context),
                    {},
                    task_id,
                )
            except Exception:  # pragma: no cover - diagnostics are best-effort
                future = self._submit_impl(func, args, kwargs, task_id)
        else:
            future = self._submit_impl(func, args, kwargs, task_id)

        def _completion_callback(task_future: TaskFuture) -> None:
            status = "completed"
            try:
                task_future.result(0)
            except TimeoutError:
                status = "timeout"
            except Exception:
                status = "failed"
            finally:
                self._on_completion(task_id, name, category, metadata, status, task_future)

        future.add_done_callback(_completion_callback)
        self._report_metrics()
        return future

    def shutdown(self) -> None:
        """Optional cleanup implemented by subclasses."""

    # ------------------------------------------------------------------
    def _on_completion(
        self,
        task_id: str,
        name: str,
        category: str,
        metadata: Dict[str, Any],
        status: str,
        task_future: TaskFuture,
    ) -> None:
        snapshot: Dict[str, Any] | None = None
        if metadata.get("autofix"):
            with self._lock:
                snapshot = self._autofix_snapshots.get(task_id)

        with self._lock:
            self._pending = max(0, self._pending - 1)
            if status == "completed":
                self._completed += 1
            else:
                self._failed += 1

        payload = {
            "task_id": task_id,
            "worker_id": self.worker_id,
            "name": name,
            "category": category,
            "status": status,
            "metadata": metadata,
        }
        if status == "completed":
            try:
                payload["result"] = task_future.result(0)
            except Exception:
                payload["result"] = None
        else:
            exc = task_future.exception(0)
            if exc is not None:
                payload["error"] = repr(exc)
                try:
                    from modules.diagnostics.auto_fixer import (
                        AutoFixFailed,
                        AutoFixer,
                        apply_fix_to_code,
                        extract_retry_kwargs,
                    )

                    if isinstance(exc, AutoFixFailed):
                        payload["autofix"] = exc.payload()
                    else:
                        autofix_cfg = metadata.get("autofix")
                        cfg = dict(autofix_cfg or {}) if isinstance(autofix_cfg, dict) else {}
                        strategy = str(cfg.get("strategy") or cfg.get("mode") or "in_task").lower()
                        llm = cfg.get("llm") if callable(cfg.get("llm")) else None

                        autofixer = AutoFixer(llm=llm) if llm is not None else AutoFixer.from_env()
                        if autofixer is not None:
                            ctx = {
                                "task_id": task_id,
                                "worker_id": self.worker_id,
                                "name": name,
                                "category": category,
                                "metadata": dict(metadata),
                            }
                            if snapshot:
                                code_arg_index = cfg.get("code_arg_index")
                                code_kwarg = cfg.get("code_kwarg")
                                code_value: str | None = None
                                snap_args = snapshot.get("args") or ()
                                snap_kwargs = snapshot.get("kwargs") or {}
                                if isinstance(code_kwarg, str) and isinstance(snap_kwargs.get(code_kwarg), str):
                                    code_value = str(snap_kwargs[code_kwarg])
                                elif isinstance(code_arg_index, int) and 0 <= code_arg_index < len(snap_args):
                                    if isinstance(snap_args[code_arg_index], str):
                                        code_value = str(snap_args[code_arg_index])
                                if code_value is not None:
                                    ctx["code"] = code_value[:12_000]
                            analysis = autofixer.analyze_error(exc, context=ctx)
                            plan = autofixer.generate_fix_plan(analysis)
                            if plan is not None:
                                payload["autofix"] = {
                                    "analysis": analysis.to_dict(),
                                    "fix_history": [plan.to_dict()],
                                }

                                if strategy == "resubmit" and snapshot and status == "failed":
                                    safety = plan.data.get("safety") if isinstance(plan.data, dict) else None
                                    safety_dict = safety if isinstance(safety, dict) else {}
                                    requires_review = bool(safety_dict.get("requires_human_review", True))
                                    risk = str(safety_dict.get("risk") or "high").strip().lower()
                                    if not requires_review and risk in {"low", "medium"}:
                                        attempt = int(metadata.get("autofix_attempt") or 0)
                                        max_attempts = int(cfg.get("max_attempts", 1))
                                        if attempt < max_attempts:
                                            new_args = tuple(snapshot.get("args") or ())
                                            new_kwargs = dict(snapshot.get("kwargs") or {})
                                            updated = False

                                            retry_kwargs = extract_retry_kwargs(plan)
                                            if retry_kwargs:
                                                new_kwargs.update(retry_kwargs)
                                                updated = True

                                            code_value: str | None = None
                                            if isinstance(code_kwarg, str) and isinstance(new_kwargs.get(code_kwarg), str):
                                                code_value = str(new_kwargs[code_kwarg])
                                            elif isinstance(code_arg_index, int) and 0 <= code_arg_index < len(new_args):
                                                if isinstance(new_args[code_arg_index], str):
                                                    code_value = str(new_args[code_arg_index])

                                            if code_value is not None:
                                                patched = apply_fix_to_code(code_value, plan)
                                                if isinstance(patched, str) and patched and patched != code_value:
                                                    if isinstance(code_kwarg, str) and isinstance(
                                                        new_kwargs.get(code_kwarg), str
                                                    ):
                                                        new_kwargs[code_kwarg] = patched
                                                        updated = True
                                                    elif isinstance(code_arg_index, int) and 0 <= code_arg_index < len(new_args):
                                                        args_list = list(new_args)
                                                        args_list[code_arg_index] = patched
                                                        new_args = tuple(args_list)
                                                        updated = True

                                            if updated:
                                                retry_task_id = uuid.uuid4().hex
                                                retry_meta = dict(metadata)
                                                retry_meta.update(
                                                    {
                                                        "task_id": retry_task_id,
                                                        "name": name,
                                                        "category": category,
                                                        "autofix_attempt": attempt + 1,
                                                        "autofix_parent_task_id": task_id,
                                                        "autofix_plan": plan.to_dict(),
                                                    }
                                                )
                                                try:
                                                    self.submit(snapshot["func"], *new_args, metadata=retry_meta, **new_kwargs)
                                                    payload["autofix_retry"] = {
                                                        "scheduled": True,
                                                        "retry_task_id": retry_task_id,
                                                        "attempt": attempt + 1,
                                                    }
                                                except Exception:  # pragma: no cover - best effort retry
                                                    pass
                except Exception:  # pragma: no cover - optional diagnostics payload
                    pass

        if snapshot is not None:
            with self._lock:
                self._autofix_snapshots.pop(task_id, None)

        self._publish_event("task.result", payload)
        self._report_metrics()

    def _publish_event(self, topic: str, event: Dict[str, Any]) -> None:
        if not self.event_bus or publish is None:
            return
        try:
            publish(self.event_bus, topic, event)
        except Exception:  # pragma: no cover - best effort
            pass

    def _report_metrics(self) -> None:
        signal = {
            "pending": float(self._pending),
            "completed": float(self._completed),
            "failed": float(self._failed),
        }
        report_resource_signal(
            self.worker_id,
            signal,
            metadata={"adapter": self.__class__.__name__},
            event_bus=self.event_bus,
        )

    def _register_worker(self) -> None:
        capabilities = {
            "adapter": self.__class__.__name__,
        }
        get_hardware_registry().register(self.worker_id, capabilities)

    # ------------------------------------------------------------------
    @abstractmethod
    def _submit_impl(
        self,
        func: TaskCallable,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
        task_id: str,
    ) -> TaskFuture:
        raise NotImplementedError


class LocalTaskAdapter(TaskAdapter):
    """Default adapter backed by a thread pool executor."""

    def __init__(
        self,
        *,
        max_workers: Optional[int] = None,
        worker_id: Optional[str] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        super().__init__(worker_id=worker_id, event_bus=event_bus)
        default_workers = max_workers or max(4, (os.cpu_count() or 4))
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=default_workers, thread_name_prefix=f"{self.worker_id}-thr"
        )

    def _submit_impl(
        self,
        func: TaskCallable,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
        task_id: str,
    ) -> TaskFuture:
        future = self._executor.submit(func, *args, **kwargs)
        return _LocalTaskFuture(future)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)


@dataclass
class _RayConfig:
    address: Optional[str] = None
    namespace: Optional[str] = None
    local_mode: bool = False


class RayTaskAdapter(TaskAdapter):
    """Adapter that dispatches tasks through Ray."""

    def __init__(
        self,
        *,
        worker_id: Optional[str] = None,
        event_bus: Optional[EventBus] = None,
        address: Optional[str] = None,
        namespace: Optional[str] = None,
        local_mode: bool = False,
    ) -> None:
        if ray is None:
            raise RuntimeError("Ray is not installed. Install ray to use RayTaskAdapter.")
        super().__init__(worker_id=worker_id, event_bus=event_bus)
        self._config = _RayConfig(address=address, namespace=namespace, local_mode=local_mode)
        if not ray.is_initialized():
            ray.init(
                address=address or None,
                namespace=namespace or "autogpt",
                include_dashboard=False,
                ignore_reinit_error=True,
                local_mode=local_mode,
            )
        self._remote_executor = ray.remote(self._execute_remote)

    @staticmethod
    def _execute_remote(func: TaskCallable, args, kwargs):
        return func(*args, **kwargs)

    def _submit_impl(
        self,
        func: TaskCallable,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
        task_id: str,
    ) -> TaskFuture:
        object_ref = self._remote_executor.remote(func, args, kwargs)
        return _RayTaskFuture(object_ref)

    def shutdown(self) -> None:
        if ray and ray.is_initialized():
            try:
                ray.shutdown()
            except Exception:  # pragma: no cover - best effort
                pass


class DaskTaskAdapter(TaskAdapter):
    """Adapter that dispatches tasks through a Dask distributed client."""

    def __init__(
        self,
        *,
        worker_id: Optional[str] = None,
        event_bus: Optional[EventBus] = None,
        address: Optional[str] = None,
    ) -> None:
        if DaskClient is None:
            raise RuntimeError("Dask distributed is not installed. Install dask[distributed] to use DaskTaskAdapter.")
        super().__init__(worker_id=worker_id, event_bus=event_bus)
        if address:
            self._client = DaskClient(address)  # pragma: no cover - requires cluster
        else:
            self._client = DaskClient(processes=True, threads_per_worker=1, dashboard_address=None)

    def _submit_impl(
        self,
        func: TaskCallable,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
        task_id: str,
    ) -> TaskFuture:
        future = self._client.submit(func, *args, **kwargs)
        return _DaskTaskFuture(future)

    def shutdown(self) -> None:
        self._client.close()


def create_task_adapter(
    mode: Optional[str] = None,
    *,
    worker_id: Optional[str] = None,
    event_bus: Optional[EventBus] = None,
    **kwargs: Any,
) -> TaskAdapter:
    """Factory that instantiates a task adapter based on configuration."""

    resolved_mode = (mode or os.getenv("TASK_ADAPTER") or "local").lower()
    if resolved_mode in {"auto", "adaptive"}:
        try:
            from modules.environment.environment_adapter import choose_task_adapter_mode

            resolved_mode = choose_task_adapter_mode()
        except Exception:
            resolved_mode = "local"
    if resolved_mode == "local":
        return LocalTaskAdapter(worker_id=worker_id, event_bus=event_bus, **kwargs)
    if resolved_mode == "ray":
        if "address" not in kwargs:
            address = os.getenv("RAY_ADDRESS") or os.getenv("RAY_HEAD_ADDRESS")
            if address:
                kwargs["address"] = address
        return RayTaskAdapter(worker_id=worker_id, event_bus=event_bus, **kwargs)
    if resolved_mode == "dask":
        if "address" not in kwargs:
            address = os.getenv("DASK_SCHEDULER_ADDRESS") or os.getenv("DASK_ADDRESS")
            if address:
                kwargs["address"] = address
        return DaskTaskAdapter(worker_id=worker_id, event_bus=event_bus, **kwargs)
    raise ValueError(f"Unsupported task adapter mode '{resolved_mode}'")
