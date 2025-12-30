"""Micro-benchmark comparing heap-based task queue vs repeated sorting."""
import heapq
import random
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "autogpts" / "autogpt"))
from third_party.autogpt.autogpt.core.planning.schema import Task, TaskType


def _make_tasks(n: int) -> list[Task]:
    return [
        Task(objective=f"task {i}", type=TaskType.TEST, priority=random.randint(1, 100), ready_criteria=[], acceptance_criteria=[])
        for i in range(n)
    ]


def benchmark(n: int = 10000) -> dict[str, float]:
    tasks = _make_tasks(n)

    start = time.perf_counter()
    heap: list[tuple[int, int, Task]] = []
    for i, t in enumerate(tasks):
        heapq.heappush(heap, (t.priority, i, t))
    while heap:
        heapq.heappop(heap)
    heap_time = time.perf_counter() - start

    start = time.perf_counter()
    queue: list[Task] = []
    for t in tasks:
        queue.append(t)
        queue.sort(key=lambda x: x.priority, reverse=True)
    while queue:
        queue.pop()
    sort_time = time.perf_counter() - start

    return {"heapq": heap_time, "sort": sort_time}


if __name__ == "__main__":
    results = benchmark()
    print(f"heapq: {results['heapq']:.4f}s, sort: {results['sort']:.4f}s")
