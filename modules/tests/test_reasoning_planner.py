"""Unit tests for the :mod:`ReasoningPlanner`."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "autogpts", "autogpt")))

from third_party.autogpt.autogpt.agent_manager import AgentManager
from third_party.autogpt.autogpt.core.planning.reasoner import ReasoningPlanner
from third_party.autogpt.autogpt.core.planning.schema import Task, TaskStatus, TaskType
from third_party.autogpt.autogpt.file_storage.local import FileStorageConfiguration, LocalFileStorage


def make_task(objective: str) -> Task:
    return Task(
        objective=objective,
        type=TaskType.PLAN,
        priority=1,
        ready_criteria=["gather requirements", "outline approach"],
        acceptance_criteria=[],
    )


def test_reasoning_planner_generates_chain_and_next_step():
    planner = ReasoningPlanner()
    task = make_task("write tests")

    result = planner.plan(task)

    assert result.next_step == task.objective
    assert result.reasoning[0] == f"Objective: {task.objective}"
    assert "Step 1" in result.reasoning[1]


def test_agent_manager_uses_reasoning_planner(tmp_path: Path):
    storage = LocalFileStorage(FileStorageConfiguration(root=tmp_path))
    planner = ReasoningPlanner()
    manager = AgentManager(storage, planner=planner)

    task = make_task("integrate planner")
    task.context.status = TaskStatus.READY

    result = manager.plan_next_step(task)

    assert result.next_step == task.objective
    assert result.reasoning[0].startswith("Objective:")

