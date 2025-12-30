import pytest
from pydantic import ValidationError

from third_party.autogpt.autogpt.core.planning.schema import Task, TaskType


def test_task_type_accepts_valid_string():
    task = Task(
        objective="Example",
        type="write",
        priority=1,
        ready_criteria=[],
        acceptance_criteria=[],
    )
    assert task.type is TaskType.WRITE


def test_task_type_rejects_invalid_string():
    with pytest.raises(ValidationError):
        Task(
            objective="Example",
            type="invalid",
            priority=1,
            ready_criteria=[],
            acceptance_criteria=[],
        )
