import sys
import types
from importlib import util
from pathlib import Path
from typing import Any, Dict, List

from modules.events import InMemoryEventBus


ROOT = Path(__file__).resolve().parents[2]

backend_pkg = types.ModuleType("backend")
backend_pkg.__path__ = [str(ROOT / "backend")]
sys.modules.setdefault("backend", backend_pkg)

execution_pkg = types.ModuleType("backend.execution")
execution_pkg.__path__ = [str(ROOT / "backend" / "execution")]
sys.modules.setdefault("backend.execution", execution_pkg)

MODULE_PATH = ROOT / "backend" / "execution" / "upgrade_decision_engine.py"
spec = util.spec_from_file_location("backend.execution.upgrade_decision_engine", MODULE_PATH)
module = util.module_from_spec(spec)
assert spec and spec.loader
sys.modules.setdefault("backend.execution.upgrade_decision_engine", module)
spec.loader.exec_module(module)

UpgradeDecisionEngine = module.UpgradeDecisionEngine


def test_upgrade_engine_requests_module_acquisition_on_repeated_cluster_failures() -> None:
    bus = InMemoryEventBus()
    requests: List[Dict[str, Any]] = []
    bus.subscribe("module.acquisition.request", lambda e: requests.append(e))

    engine = UpgradeDecisionEngine(
        event_bus=bus,  # type: ignore[arg-type]
        enabled=True,
        cooldown_secs=0.0,
        failure_threshold=999,
        same_cluster_threshold=2,
        learning_enabled=False,
    )

    bus.publish("task_manager.task_completed", {"status": "failed", "error": "Cannot open image file"})
    bus.publish("task_manager.task_completed", {"status": "failed", "error": "image decode failed"})
    bus.join()

    assert requests
    assert "query" in requests[0]
    assert "vision" in str(requests[0]["query"])

    engine.close()


def test_upgrade_engine_requests_architecture_when_success_rate_stagnates_below_target() -> None:
    bus = InMemoryEventBus()
    requests: List[Dict[str, Any]] = []
    bus.subscribe("upgrade.architecture.request", lambda e: requests.append(e))

    engine = UpgradeDecisionEngine(
        event_bus=bus,  # type: ignore[arg-type]
        enabled=True,
        cooldown_secs=0.0,
        failure_threshold=999,
        same_cluster_threshold=999,
        stagnation_window=3,
        stagnation_min_delta=0.02,
        success_rate_target=0.9,
        learning_enabled=False,
    )

    for _ in range(6):
        bus.publish("task_manager.task_completed", {"status": "failed", "error": "generic failure"})
    bus.join()

    assert requests
    assert requests[0].get("source") == "upgrade_decision"

    engine.close()
