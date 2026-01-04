import logging
import sys
from importlib import util
from pathlib import Path

import pytest

from modules.events import InMemoryEventBus


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_simple_brain_adapter_runtime_failover(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRAIN_RUNTIME_FAILOVER_ENABLED", "1")
    monkeypatch.setenv("BRAIN_RUNTIME_RESTART_ON_FAILURE", "1")
    monkeypatch.setenv("BRAIN_RUNTIME_FAILOVER_THRESHOLD", "2")
    monkeypatch.setenv("BRAIN_RUNTIME_FAILOVER_WINDOW_SECS", "60")
    monkeypatch.setenv("BRAIN_RUNTIME_FAILOVER_COOLDOWN_SECS", "0")

    from autogpt.core.brain.config import BrainBackend

    cognition_path = (
        ROOT / "third_party" / "autogpt" / "autogpt" / "core" / "agent" / "cognition.py"
    )
    spec = util.spec_from_file_location("unit_test.autogpt.core.agent.cognition", cognition_path)
    cognition = util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(cognition)

    class FailingBrain:
        def process_cycle(self, payload):
            raise RuntimeError("boom")

        def shutdown(self) -> None:
            return

    class WorkingBrain:
        def process_cycle(self, payload):
            return {"ok": True}

    def fake_create_brain_backend(backend, **kwargs):
        if backend == BrainBackend.BRAIN_SIMULATION:
            return FailingBrain()
        if backend == BrainBackend.WHOLE_BRAIN:
            return WorkingBrain()
        raise AssertionError(f"Unexpected backend: {backend!r}")

    monkeypatch.setattr(cognition, "create_brain_backend", fake_create_brain_backend)

    bus = InMemoryEventBus()
    failures = []
    failovers = []
    restarts = []
    bus.subscribe("brain.backend.failure", lambda e: failures.append(e))
    bus.subscribe("brain.backend.failover", lambda e: failovers.append(e))
    bus.subscribe("brain.backend.restart", lambda e: restarts.append(e))

    adapter = cognition.SimpleBrainAdapter(
        cognition.SimpleBrainAdapter.default_settings,
        logging.getLogger("test_simple_brain_adapter_runtime_failover"),
        event_bus=bus,
    )

    result = adapter._process_cycle({"text": "hi"}, callsite="unit-test")
    bus.join()

    assert result == {"ok": True}
    assert adapter._backend == BrainBackend.WHOLE_BRAIN
    assert len(failures) >= 2
    assert restarts
    assert failovers
    assert failovers[-1]["to_backend"] == BrainBackend.WHOLE_BRAIN.value
