import time
from typing import Any, Dict, List

from modules.events import InMemoryEventBus

from backend.execution.task_manager import TaskManager, TaskPriority
from backend.execution.task_submission_scheduler import TaskSubmissionScheduler


def test_task_submission_scheduler_infers_priority_and_emits_status() -> None:
    bus = InMemoryEventBus()
    statuses: List[Dict[str, Any]] = []
    bus.subscribe("scheduler.status", lambda e: statuses.append(e))

    task_manager = TaskManager(event_bus=bus, default_cpu_workers=2)
    scheduler = TaskSubmissionScheduler(
        task_manager=task_manager,
        event_bus=bus,
        enabled=True,
        emit_status_on_submit=True,
    )
    try:
        def work() -> str:
            time.sleep(0.01)
            return "ok"

        handle = scheduler.submit_task(
            work,
            priority=None,  # auto
            category="planning",
            name="test_planning",
            metadata={"interactive": True},
        )
        bus.join()

        assert int(handle.priority) >= int(TaskPriority.HIGH)
        meta = handle.metadata
        assert isinstance(meta.get("scheduler"), dict)
        assert int(meta["scheduler"]["priority"]) >= int(TaskPriority.HIGH)
        assert statuses
        assert any(item.get("trigger") == "task_scheduled" for item in statuses)
    finally:
        task_manager.shutdown()

