"""Simple distributed task execution using multiprocessing managers.

The :class:`DistributedBrainNode` enables basic task distribution across
processes or machines. It uses Python's ``multiprocessing.managers`` module to
expose task and result queues over a network connection.  Worker nodes request
work from the task queue, execute it locally and send the results back to the
result queue.  The master node can then aggregate the results once all tasks
have been processed.

This module intentionally keeps the implementation lightweight and free from
external dependencies so that it can run in constrained environments.  It can
serve as a foundation for more sophisticated message queue or RPC based
approaches (e.g. gRPC) in the future.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass, field
from multiprocessing.managers import SyncManager
from typing import Any, Callable, Iterable, List, Tuple


# ----------------------------------------------------------------------------
# Manager utilities
# ----------------------------------------------------------------------------

class _QueueManager(SyncManager):
    """Private manager class used to expose shared queues."""


@dataclass
class DistributedBrainNode:
    """Utility for distributing tasks and aggregating results.

    Parameters
    ----------
    address: Tuple[str, int]
        ``(host, port)`` pair used by the manager process.
    authkey: bytes
        Authentication key required to connect to the manager.
    """

    address: Tuple[str, int] = ("localhost", 50000)
    authkey: bytes = b"autogpt"
    manager: _QueueManager | None = field(init=False, default=None)
    task_queue: "queue.Queue[Any]" | None = field(init=False, default=None)
    result_queue: "queue.Queue[Any]" | None = field(init=False, default=None)

    # ------------------------------------------------------------------
    # Master side API
    # ------------------------------------------------------------------
    def start_master(self, tasks: Iterable[Any]) -> None:
        """Start a manager and enqueue the provided ``tasks``.

        This method must be called on the master node.  It starts a
        ``SyncManager`` in a background process exposing two queues: one for
        tasks and one for results.  The supplied tasks are pushed onto the
        task queue so that worker nodes can retrieve them.
        """

        task_q: "queue.Queue[Any]" = queue.Queue()
        result_q: "queue.Queue[Any]" = queue.Queue()

        _QueueManager.register("get_task_queue", callable=lambda: task_q)
        _QueueManager.register("get_result_queue", callable=lambda: result_q)

        self.manager = _QueueManager(address=self.address, authkey=self.authkey)
        self.manager.start()

        # Obtain proxied queues from the manager process
        self.task_queue = self.manager.get_task_queue()  # type: ignore[assignment]
        self.result_queue = self.manager.get_result_queue()  # type: ignore[assignment]

        for task in tasks:
            self.task_queue.put(task)

    def gather_results(self, expected_results: int) -> List[Any]:
        """Collect ``expected_results`` items from the result queue."""

        if not self.result_queue:
            raise RuntimeError("Master has not been started.")

        results: List[Any] = []
        for _ in range(expected_results):
            try:
                results.append(self.result_queue.get(timeout=5))
            except queue.Empty:
                results.append(None)
        return results

    def shutdown(self) -> None:
        """Shut down the manager process if running."""

        if self.manager:
            self.manager.shutdown()
            self.manager = None
            self.task_queue = None
            self.result_queue = None

    # ------------------------------------------------------------------
    # Worker side API
    # ------------------------------------------------------------------
    def run_worker(self, handler: Callable[[Any], Any]) -> None:
        """Run a worker loop using ``handler`` to process tasks.

        The worker connects to the master's manager, pulls tasks from the task
        queue and pushes results onto the result queue.  The loop exits when no
        more tasks are available.
        """

        _QueueManager.register("get_task_queue")
        _QueueManager.register("get_result_queue")
        manager = _QueueManager(address=self.address, authkey=self.authkey)
        manager.connect()
        task_q = manager.get_task_queue()
        result_q = manager.get_result_queue()

        while True:
            try:
                task = task_q.get(timeout=1)
            except queue.Empty:
                break
            try:
                result_q.put(handler(task))
            except Exception as err:  # pragma: no cover - best effort logging
                result_q.put(err)
