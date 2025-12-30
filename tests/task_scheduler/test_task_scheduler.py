"""Tests for TaskScheduler."""

from modules.task_scheduler import TaskScheduler


def test_scheduler_orders_and_retries():
    scheduler = TaskScheduler()
    scheduler.enqueue(
        [
            {"description": "low", "priority": 0.1},
            {"description": "high", "priority": 0.9},
        ]
    )
    first = scheduler.next_task()
    assert first.description == "high"
    scheduler.report_failure(first)
    # After one retry still in queue unless exceeded max_retries.
    next_task = scheduler.next_task()
    assert next_task.description == "high"
