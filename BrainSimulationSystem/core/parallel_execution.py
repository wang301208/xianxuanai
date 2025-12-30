"""
Parallel execution utilities for region-level simulation updates.

The executor abstraction allows BrainRegionNetwork to run per-region
updates on multiple worker threads or worker processes. Process-based
execution requires picklable runners/inputs and should avoid reliance on
shared mutable state because each worker receives a serialized copy.
"""

from __future__ import annotations

import concurrent.futures
import copy
import logging
import pickle
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RegionUpdateTask:
    """Container describing a single region update invocation."""

    name: Any
    runner: Callable[[float, Dict[str, Any]], Dict[str, Any]]
    dt: float
    inputs: Dict[str, Any]
    mode: str


class RegionParallelExecutor:
    """Coordinates region update execution across multiple workers."""

    _SUPPORTED_MODES = {"auto", "serial", "thread", "process", "distributed"}

    def __init__(
        self,
        strategy: str = "auto",
        max_workers: Optional[int] = None,
        distributed: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._strategy = "auto"
        self._max_workers = max_workers
        self._distributed_cfg = distributed or {}
        self._thread_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._process_pool: Optional[concurrent.futures.ProcessPoolExecutor] = None
        self._pool_lock = threading.Lock()
        self._thread_pool_size: Optional[int] = None
        self._process_pool_size: Optional[int] = None
        self.configure(strategy=strategy, max_workers=max_workers, distributed=distributed)

    # Public API -----------------------------------------------------------------
    def configure(
        self,
        strategy: Optional[str] = None,
        max_workers: Optional[int] = None,
        distributed: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update execution policy. Existing pools are recycled when possible."""

        if strategy is not None:
            normalized = strategy.lower()
            if normalized not in self._SUPPORTED_MODES:
                raise ValueError(f"Unsupported parallel strategy '{strategy}'")
            self._strategy = normalized
        if max_workers is not None:
            self._max_workers = max_workers
        if distributed is not None:
            if isinstance(distributed, bool):
                self._distributed_cfg = {"enabled": distributed}
            else:
                self._distributed_cfg = distributed

        if self._strategy == "distributed" and not self._distributed_enabled():
            self._logger.warning(
                "Distributed mode requested without enabling configuration; reverting to 'auto'."
            )
            self._strategy = "auto"

    def run(self, tasks: Sequence[RegionUpdateTask]) -> Dict[Any, Dict[str, Any]]:
        """Execute the provided region update tasks according to the configured policy."""

        if not tasks:
            return {}

        strategy = self._effective_strategy(len(tasks))

        if strategy == "serial":
            return self._run_serial(tasks)
        if strategy == "thread":
            return self._run_threaded(tasks)
        if strategy == "process":
            return self._run_process(tasks)
        if strategy == "distributed":
            return self._run_process(tasks)

        # Default backstop
        return self._run_serial(tasks)

    def shutdown(self) -> None:
        """Release any pooled resources."""
        with self._pool_lock:
            if self._thread_pool is not None:
                self._thread_pool.shutdown(wait=True, cancel_futures=True)
                self._thread_pool = None
                self._thread_pool_size = None
            if self._process_pool is not None:
                self._process_pool.shutdown(wait=True, cancel_futures=True)
                self._process_pool = None
                self._process_pool_size = None

    # Internal helpers -----------------------------------------------------------
    def _effective_strategy(self, num_tasks: int) -> str:
        # If the caller explicitly requests a strategy (e.g. "process"), honor it
        # even for a single task so we can validate requirements like picklability.
        if num_tasks <= 1:
            if self._strategy == "auto":
                return "serial"
            return self._strategy
        if self._strategy == "auto":
            return "thread"
        return self._strategy

    def _distributed_enabled(self) -> bool:
        if isinstance(self._distributed_cfg, dict):
            enabled = self._distributed_cfg.get("enabled", False)
            return bool(enabled)
        return bool(self._distributed_cfg)

    def _run_serial(self, tasks: Sequence[RegionUpdateTask]) -> Dict[Any, Dict[str, Any]]:
        return dict(self._execute(task) for task in tasks)

    def _run_threaded(self, tasks: Sequence[RegionUpdateTask]) -> Dict[Any, Dict[str, Any]]:
        if len(tasks) == 1:
            return self._run_serial(tasks)

        pool = self._ensure_thread_pool(len(tasks))
        futures = {pool.submit(self._execute, task): task.name for task in tasks}
        results: Dict[Any, Dict[str, Any]] = {}
        for future in concurrent.futures.as_completed(futures):
            name, payload = future.result()
            results[name] = payload
        return results

    def _run_process(self, tasks: Sequence[RegionUpdateTask]) -> Dict[Any, Dict[str, Any]]:
        """Execute tasks in a separate process pool with lifecycle management."""

        if len(tasks) == 1:
            # Validate picklability/deepcopy constraints even when we run a single task
            # inline to keep behavior consistent with process strategy expectations.
            self._prepare_process_payload(tasks[0])
            return self._run_serial(tasks)

        serialized_tasks = [self._prepare_process_payload(task) for task in tasks]
        pool = self._ensure_process_pool(len(serialized_tasks))
        futures = {pool.submit(_process_worker, payload): payload[0] for payload in serialized_tasks}
        results: Dict[Any, Dict[str, Any]] = {}
        for future in concurrent.futures.as_completed(futures):
            name, payload = future.result()
            results[name] = payload
        return results

    def _ensure_thread_pool(self, workload_size: int) -> concurrent.futures.ThreadPoolExecutor:
        desired = self._max_workers or workload_size
        with self._pool_lock:
            if self._thread_pool is None or self._thread_pool_size != desired:
                if self._thread_pool is not None:
                    self._thread_pool.shutdown(wait=True, cancel_futures=True)
                    self._thread_pool_size = None
                self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=desired)
                self._thread_pool_size = desired
        return self._thread_pool

    def _ensure_process_pool(
        self, workload_size: int
    ) -> concurrent.futures.ProcessPoolExecutor:
        desired = self._max_workers or workload_size
        with self._pool_lock:
            if self._process_pool is None or self._process_pool_size != desired:
                if self._process_pool is not None:
                    self._process_pool.shutdown(wait=True, cancel_futures=True)
                    self._process_pool_size = None
                self._process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=desired)
                self._process_pool_size = desired
        assert self._process_pool is not None  # for type checkers
        return self._process_pool

    @staticmethod
    def _execute(task: RegionUpdateTask) -> Tuple[Any, Dict[str, Any]]:
        import time

        start = time.perf_counter()
        result = task.runner(task.dt, task.inputs)
        elapsed = time.perf_counter() - start
        # Defensive copy to avoid upstream mutation from worker threads.
        payload = dict(result)
        payload["mode"] = task.mode
        payload.setdefault("elapsed", elapsed)
        return task.name, payload

    @staticmethod
    def _prepare_process_payload(task: RegionUpdateTask) -> Tuple[Any, Callable, float, Dict[str, Any], str]:
        try:
            inputs = copy.deepcopy(task.inputs)
        except Exception as exc:  # pragma: no cover - exercised in tests via ValueError
            raise ValueError(
                "RegionUpdateTask.inputs must be deepcopy-able for process execution"
            ) from exc

        try:
            pickle.dumps(task.runner)
            pickle.dumps(inputs)
        except Exception as exc:  # pragma: no cover - exercised in tests via ValueError
            raise ValueError(
                "Process-based execution requires picklable runner and inputs"
            ) from exc

        return task.name, task.runner, task.dt, inputs, task.mode


def _process_worker(payload: Tuple[Any, Callable, float, Dict[str, Any], str]) -> Tuple[Any, Dict[str, Any]]:
    import time

    name, runner, dt, inputs, mode = payload
    start = time.perf_counter()
    result = runner(dt, inputs)
    elapsed = time.perf_counter() - start
    payload = dict(result)
    payload["mode"] = mode
    payload.setdefault("elapsed", elapsed)
    return name, payload


__all__ = ["RegionUpdateTask", "RegionParallelExecutor"]
