import heapq
from autogpt.core.agent.simple import SimpleAgent
from autogpt.core.planning.schema import Task, TaskType


def _dummy_agent() -> SimpleAgent:
    agent = SimpleAgent.__new__(SimpleAgent)
    agent._task_queue = []  # type: ignore[attr-defined]
    return agent


def _make_task(priority: int) -> Task:
    return Task(
        objective=f"task {priority}",
        type=TaskType.TEST,
        priority=priority,
        ready_criteria=[],
        acceptance_criteria=[],
    )


def test_route_task_maintains_priority_order():
    agent = _dummy_agent()
    agent.route_task(_make_task(3))
    agent.route_task(_make_task(1))
    agent.route_task(_make_task(2))

    priorities = [heapq.heappop(agent._task_queue)[1].priority for _ in range(len(agent._task_queue))]
    assert priorities == [1, 2, 3]
