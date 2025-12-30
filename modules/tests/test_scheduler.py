import os
import sys
import asyncio

sys.path.insert(0, os.path.abspath(os.getcwd()))

from backend.execution.scheduler import Scheduler
from backend.execution.task_graph import TaskGraph


def test_pick_least_busy_distributes_tasks():
    scheduler = Scheduler(weights={"cpu": 0.7, "memory": 0.2, "tasks": 0.1})
    scheduler.add_agent("A")
    scheduler.add_agent("B")
    scheduler.add_agent("C")

    scheduler.update_agent("A", cpu=0.5, memory=0.1)
    scheduler.update_agent("B", cpu=0.4, memory=0.3)
    scheduler.update_agent("C", cpu=0.2, memory=0.9)

    order = []
    for _ in range(3):
        agent = scheduler._pick_least_busy()
        order.append(agent)
        scheduler._update_tasks(agent, 1)

    assert order == ["C", "B", "A"]


def test_task_weight_affects_selection():
    scheduler = Scheduler(weights={"cpu": 1.0, "memory": 1.0, "tasks": 2.0})
    scheduler.add_agent("A")
    scheduler.add_agent("B")

    scheduler.update_agent("A", cpu=0.1, memory=0.1)
    scheduler.update_agent("B", cpu=0.1, memory=0.1)

    scheduler._update_tasks("A", 2)

    assert scheduler._pick_least_busy() == "B"


def test_submit_tracks_per_agent_task_counts():
    scheduler = Scheduler()
    scheduler.add_agent("A")
    scheduler.add_agent("B")

    graph = TaskGraph()
    for i in range(4):
        graph.add_task(f"t{i}", description="task", skill=f"s{i}")

    def worker(agent: str, skill: str) -> str:
        return agent

    results = asyncio.run(scheduler.submit(graph, worker))
    assert set(results.keys()) == {f"t{i}" for i in range(4)}
    counts = scheduler.task_counts()
    assert counts["A"] + counts["B"] == 4
    assert counts["A"] > 0 and counts["B"] > 0


def test_high_concurrency_balanced_distribution():
    scheduler = Scheduler()
    for name in ["A", "B", "C", "D"]:
        scheduler.add_agent(name)

    graph = TaskGraph()
    for i in range(100):
        graph.add_task(f"t{i}", description="task", skill=f"s{i}")

    def worker(agent: str, skill: str) -> str:
        return agent

    asyncio.run(scheduler.submit(graph, worker))
    counts = scheduler.task_counts()
    assert sum(counts.values()) == 100
    assert max(counts.values()) - min(counts.values()) <= 2
