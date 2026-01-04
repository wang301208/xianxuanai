import logging
import sys
from importlib import util
from pathlib import Path

import pytest

from modules.events import InMemoryEventBus


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_brain_adapter_handles_scheduler_control_update_config(monkeypatch: pytest.MonkeyPatch) -> None:
    cognition_path = (
        ROOT / "third_party" / "autogpt" / "autogpt" / "core" / "agent" / "cognition.py"
    )
    spec = util.spec_from_file_location("unit_test.autogpt.core.agent.cognition", cognition_path)
    cognition = util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(cognition)

    class DummyBrain:
        def __init__(self) -> None:
            self.update_calls: list[dict] = []

        def update_config(self, runtime_config=None, *, overrides=None) -> None:
            self.update_calls.append({"runtime_config": runtime_config, "overrides": overrides})

    dummy = DummyBrain()

    def fake_create_brain_backend(backend, **kwargs):
        return dummy

    monkeypatch.setattr(cognition, "create_brain_backend", fake_create_brain_backend)

    bus = InMemoryEventBus()
    adapter = cognition.SimpleBrainAdapter(
        cognition.SimpleBrainAdapter.default_settings,
        logging.getLogger("test_brain_adapter_scheduler_control"),
        event_bus=bus,
    )
    adapter._agent_id = "alpha"

    bus.publish(
        "scheduler.control",
        {"action": "brain.update_config", "agent_id": "alpha", "overrides": {"dt": 123}},
    )
    bus.publish(
        "scheduler.control",
        {"action": "brain.update_config", "agent_id": "beta", "overrides": {"dt": 456}},
    )
    bus.join()

    assert dummy.update_calls
    assert dummy.update_calls[0]["overrides"]["dt"] == 123
    assert len(dummy.update_calls) == 1


def test_brain_adapter_handles_scheduler_control_switch_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    cognition_path = (
        ROOT / "third_party" / "autogpt" / "autogpt" / "core" / "agent" / "cognition.py"
    )
    spec = util.spec_from_file_location("unit_test.autogpt.core.agent.cognition_switch", cognition_path)
    cognition = util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(cognition)

    class DummyBrain:
        def update_config(self, runtime_config=None, *, overrides=None) -> None:  # pragma: no cover - unused
            return

    def fake_create_brain_backend(backend, **kwargs):
        return DummyBrain()

    monkeypatch.setattr(cognition, "create_brain_backend", fake_create_brain_backend)

    bus = InMemoryEventBus()
    adapter = cognition.SimpleBrainAdapter(
        cognition.SimpleBrainAdapter.default_settings,
        logging.getLogger("test_brain_adapter_scheduler_control_switch"),
        event_bus=bus,
    )
    adapter._agent_id = "alpha"

    bus.publish(
        "scheduler.control",
        {"action": "brain.switch_backend", "agent_id": "alpha", "backend": "WHOLE_BRAIN", "reason": "test"},
    )
    bus.join()

    assert adapter._backend == cognition.BrainBackend.WHOLE_BRAIN
