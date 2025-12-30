import asyncio
import sys
from types import ModuleType
from unittest.mock import Mock


def test_handle_result_with_non_mapping_result(monkeypatch):
    import modules as modules_pkg

    numpy_stub = ModuleType("numpy")
    monkeypatch.setitem(sys.modules, "numpy", numpy_stub)

    env_stub = ModuleType("modules.environment")

    class _DummyRegistry:
        def list_services(self):
            return {}

    def _subscribe(*args, **kwargs):
        return lambda: None

    env_stub.dispatch_task = lambda *args, **kwargs: None
    env_stub.get_hardware_registry = lambda: _DummyRegistry()
    env_stub.subscribe_resource_signals = _subscribe
    env_stub.subscribe_service_catalog = _subscribe
    env_stub.subscribe_service_signals = _subscribe
    env_stub.subscribe_task_dispatch = _subscribe
    env_stub.subscribe_task_results = _subscribe
    monkeypatch.setitem(sys.modules, "modules.environment", env_stub)
    monkeypatch.setattr(modules_pkg, "environment", env_stub, raising=False)

    from modules.pipeline.scheduler import PipelineScheduler

    scheduler = PipelineScheduler(event_bus=None, pipeline=["stage1", "stage2"])
    scheduler._inflight["task-1"] = {"stage": "stage1", "service_id": "svc-1"}

    schedule_mock = Mock()
    monkeypatch.setattr(scheduler, "schedule_stage", schedule_mock)

    event = {"task_id": "task-1", "result": "not-a-mapping"}

    asyncio.run(scheduler._handle_result(event))

    assert "task-1" not in scheduler._inflight
    schedule_mock.assert_not_called()
