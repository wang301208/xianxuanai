"""Simple priority-based task scheduler with retries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import heapq
import time


@dataclass(order=True)
class ScheduledTask:
    sort_index: float
    priority: float = field(compare=False)
    description: str = field(compare=False)
    reason: str = field(compare=False, default="curiosity")
    retries: int = field(compare=False, default=0)
    max_retries: int = field(compare=False, default=3)
    metadata: Dict[str, Any] = field(compare=False, default_factory=dict)


class TaskScheduler:
    """Priority queue for autonomous tasks with retry/abandon semantics."""

    def __init__(self) -> None:
        self._queue: List[ScheduledTask] = []
        self._abandoned: List[ScheduledTask] = []

    def enqueue(self, tasks: List[Dict[str, Any]]) -> None:
        now = time.time()
        for task in tasks:
            priority = float(task.get("priority", 0.5))
            description = task.get("description", "autonomous-task")
            scheduled = ScheduledTask(
                sort_index=-(priority + now * 0.0),  # negative for max-heap behavior with heapq
                priority=priority,
                description=description,
                reason=task.get("reason", "autonomous"),
                retries=int(task.get("retries", 0)),
                max_retries=int(task.get("max_retries", 3)),
                metadata=dict(task.get("metadata", {})),
            )
            heapq.heappush(self._queue, scheduled)

    def next_task(self) -> Optional[ScheduledTask]:
        """Pop the highest-priority task, if any."""

        if not self._queue:
            return None
        return heapq.heappop(self._queue)

    def report_failure(self, task: ScheduledTask) -> None:
        """Requeue or abandon based on retries."""

        task.retries += 1
        if task.retries > task.max_retries:
            self._abandoned.append(task)
            return
        task.sort_index = -(task.priority - 0.05 * task.retries)
        heapq.heappush(self._queue, task)

    def abandoned(self) -> List[ScheduledTask]:
        return list(self._abandoned)
