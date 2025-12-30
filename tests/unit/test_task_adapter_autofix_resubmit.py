from __future__ import annotations

import time

import pytest

from modules.events import InMemoryEventBus
from modules.execution.task_adapter import LocalTaskAdapter


def test_task_adapter_autofix_resubmits_failed_task() -> None:
    bus = InMemoryEventBus()
    events: list[dict] = []

    async def _handler(event: dict) -> None:
        events.append(event)

    bus.subscribe("task.result", _handler)

    def llm_stub(_: str) -> str:
        return (
            "```json\n"
            '{\n  "analysis": {"likely_root_cause": "division by zero", "confidence": 0.9},\n'
            '  "fix": {"kind": "retry_kwargs", "retry_kwargs": {"b": 1.0}},\n'
            '  "safety": {"requires_human_review": false, "risk": "low"}\n'
            "}\n"
            "```"
        )

    def divide(a: float, *, b: float) -> float:
        return a / b

    adapter = LocalTaskAdapter(worker_id="test-adapter", event_bus=bus, max_workers=1)
    try:
        future = adapter.submit(
            divide,
            2.0,
            b=0.0,
            metadata={
                "task_id": "root",
                "name": "divide",
                "category": "unit",
                "autofix": {"strategy": "resubmit", "max_attempts": 1, "llm": llm_stub},
            },
        )

        with pytest.raises(ZeroDivisionError):
            future.result(timeout=2)

        deadline = time.time() + 5.0
        retry_task_id: str | None = None
        while time.time() < deadline:
            bus.join()
            for ev in list(events):
                if ev.get("task_id") == "root" and ev.get("status") == "failed":
                    retry_info = ev.get("autofix_retry") or {}
                    retry_task_id = retry_info.get("retry_task_id")
            if retry_task_id:
                break
            time.sleep(0.05)

        assert retry_task_id, "expected AutoFix resubmit to schedule a retry task id"

        deadline = time.time() + 5.0
        completed = False
        while time.time() < deadline:
            bus.join()
            for ev in list(events):
                if ev.get("task_id") == retry_task_id and ev.get("status") == "completed":
                    completed = True
            if completed:
                break
            time.sleep(0.05)

        assert completed, "expected retry task to complete successfully"
    finally:
        adapter.shutdown()

