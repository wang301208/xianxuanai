from __future__ import annotations

import asyncio
import heapq
import os
import threading
import inspect
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .task_graph import TaskGraph


class Scheduler:
    """Dispatch tasks to the least busy agents based on resource usage."""

    def __init__(
        self,
        task_callback: Optional[Callable[[int], None]] = None,
        *,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        env_weights = {
            "cpu": float(os.getenv("SCHEDULER_CPU_WEIGHT", 1.0)),
            "memory": float(os.getenv("SCHEDULER_MEMORY_WEIGHT", 1.0)),
            "tasks": float(os.getenv("SCHEDULER_TASK_WEIGHT", 1.0)),
        }
        if weights:
            env_weights.update(weights)
        self._weights = env_weights
        self._agents: Dict[str, Dict[str, float]] = {}
        # Track total completed tasks for fairness verification
        self._task_counts: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._task_callback = task_callback
        # Heap of (score, revision, name) for O(log n) selection
        self._heap: List[tuple[float, int, str]] = []
        self._revisions: Dict[str, int] = {}

    def set_task_callback(self, cb: Callable[[int], None]) -> None:
        """Set a callback to be notified when task counts change."""
        self._task_callback = cb

    # ------------------------------------------------------------------
    # Agent management API
    # ------------------------------------------------------------------
    def add_agent(self, name: str) -> None:
        """Register a new agent with default utilization."""
        with self._lock:
            if name in self._agents:
                return
            self._agents[name] = {"cpu": 0.0, "memory": 0.0, "tasks": 0.0}
            self._task_counts[name] = 0
            self._push(name)

    def remove_agent(self, name: str) -> None:
        """Remove an agent from scheduling."""
        with self._lock:
            self._agents.pop(name, None)
            self._task_counts.pop(name, None)
            self._revisions.pop(name, None)

    def update_agent(self, name: str, cpu: float, memory: float) -> None:
        """Update utilization metrics for an agent."""
        with self._lock:
            if name in self._agents:
                self._agents[name]["cpu"] = cpu
                self._agents[name]["memory"] = memory
                self._push(name)

    def _score(self, name: str) -> float:
        metrics = self._agents[name]
        return (
            self._weights["cpu"] * metrics.get("cpu", 0.0)
            + self._weights["memory"] * metrics.get("memory", 0.0)
            + self._weights["tasks"] * metrics.get("tasks", 0.0)
        )

    def _push(self, name: str) -> None:
        rev = self._revisions.get(name, 0) + 1
        self._revisions[name] = rev
        heapq.heappush(self._heap, (self._score(name), rev, name))

    def _update_tasks(self, name: str, delta: float) -> None:
        self._agents[name]["tasks"] += delta
        self._push(name)

    def set_weights(self, **weights: float) -> None:
        """Update metric weights used when scoring agents for dispatch."""

        with self._lock:
            updated = False
            for key in ("cpu", "memory", "tasks"):
                if key in weights:
                    try:
                        self._weights[key] = float(weights[key])
                        updated = True
                    except (TypeError, ValueError):
                        continue
            if not updated:
                return
            for name in self._agents:
                self._push(name)

    # ------------------------------------------------------------------
    def _pick_least_busy(self) -> str | None:
        with self._lock:
            while self._heap:
                score, rev, name = heapq.heappop(self._heap)
                if name not in self._agents:
                    continue
                if self._revisions.get(name) != rev:
                    continue
                return name
            return None

    # ------------------------------------------------------------------
    async def submit(
        self, graph: TaskGraph, worker: Callable[[str, str], Awaitable[Any]] | Callable[[str, str], Any]
    ) -> Dict[str, Any]:
        """Schedule tasks on available agents and execute them in parallel.

        The ``worker`` coroutine receives the selected ``agent`` name and the
        ``skill`` associated with each task, allowing downstream consumers to
        route execution appropriately.
        """
        indegree: Dict[str, int] = {}
        dependents: Dict[str, List[str]] = {}
        for task_id, task in graph.tasks.items():
            indegree[task_id] = len(task.dependencies)
            for dep in task.dependencies:
                dependents.setdefault(dep, []).append(task_id)

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        for tid, deg in indegree.items():
            if deg == 0:
                queue.put_nowait(tid)
        if self._task_callback:
            self._task_callback(queue.qsize())

        results: Dict[str, Any] = {}
        max_workers = max(len(self._agents), 1)

        async def worker_loop() -> None:
            try:
                while True:
                    task_id = await queue.get()
                    if task_id is None:
                        queue.task_done()
                        break
                    task = graph.tasks[task_id]
                    if not task.skill:
                        results[task_id] = None
                        queue.task_done()
                        continue
                    agent = self._pick_least_busy()
                    if agent is None:
                        queue.put_nowait(task_id)
                        queue.task_done()
                        await asyncio.sleep(0)
                        continue
                    try:
                        with self._lock:
                            self._update_tasks(agent, 1)
                        try:
                            res = worker(agent, task.skill)
                            if inspect.isawaitable(res):
                                res = await res
                            results[task_id] = res
                            with self._lock:
                                self._task_counts[agent] += 1
                            for dep in dependents.get(task_id, []):
                                indegree[dep] -= 1
                                if indegree[dep] == 0:
                                    queue.put_nowait(dep)
                            if self._task_callback:
                                self._task_callback(queue.qsize())
                        except Exception as exc:
                            results[task_id] = exc
                    finally:
                        with self._lock:
                            self._update_tasks(agent, -1)
                        queue.task_done()
            except asyncio.CancelledError:
                raise

        workers = [asyncio.create_task(worker_loop()) for _ in range(max_workers)]
        try:
            await queue.join()
            for _ in workers:
                queue.put_nowait(None)
            await asyncio.gather(*workers)
        except asyncio.CancelledError:
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise

        if self._task_callback:
            self._task_callback(0)
        return {tid: results.get(tid) for tid in graph.execution_order()}

    # ------------------------------------------------------------------
    def task_counts(self) -> Dict[str, int]:
        """Return a snapshot of total completed tasks per agent."""
        with self._lock:
            return dict(self._task_counts)


__all__ = ["Scheduler"]
