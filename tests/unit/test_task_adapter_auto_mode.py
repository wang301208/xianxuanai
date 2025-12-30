from __future__ import annotations

from modules.execution.task_adapter import LocalTaskAdapter, create_task_adapter


def test_create_task_adapter_auto_defaults_to_local(monkeypatch) -> None:
    monkeypatch.delenv("TASK_ADAPTER", raising=False)
    adapter = create_task_adapter("auto")
    assert isinstance(adapter, LocalTaskAdapter)
    adapter.shutdown()

