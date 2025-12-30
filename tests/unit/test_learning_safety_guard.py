import sys
import types
from importlib import util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
backend_pkg = types.ModuleType("backend")
backend_pkg.__path__ = [str(ROOT / "backend")]
sys.modules.setdefault("backend", backend_pkg)
execution_pkg = types.ModuleType("backend.execution")
execution_pkg.__path__ = [str(ROOT / "backend" / "execution")]
sys.modules.setdefault("backend.execution", execution_pkg)

MODULE_PATH = ROOT / "backend" / "execution" / "learning_safety.py"
spec = util.spec_from_file_location("backend.execution.learning_safety", MODULE_PATH)
learning_safety = util.module_from_spec(spec)
assert spec and spec.loader
sys.modules.setdefault("backend.execution.learning_safety", learning_safety)
spec.loader.exec_module(learning_safety)
LearningSafetyGuard = learning_safety.LearningSafetyGuard


class _DummyModel:
    def __init__(self, value: int) -> None:
        self.value = int(value)

    def get_state(self):
        return {"value": int(self.value)}

    def set_state(self, state):
        self.value = int(state["value"])


class _DummyLearningManager:
    def __init__(self) -> None:
        self.pauses = []

    def pause(self, seconds: float, *, reason: str = "paused") -> None:
        self.pauses.append((float(seconds), str(reason)))


def test_learning_safety_guard_rolls_back_on_regression() -> None:
    guard = LearningSafetyGuard(
        enabled=True,
        patience=3,
        min_delta=0.0,
        rollback_delta=0.1,
        pause_seconds=10.0,
        monitor_window=1,
    )
    model = _DummyModel(1)
    imitation = _DummyModel(7)
    manager = _DummyLearningManager()

    first = guard.evaluate(
        stats={"cross_domain_success": 0.6},
        performance_monitor=None,
        predictive_model=model,
        imitation_policy=imitation,
        learning_manager=manager,
        event_bus=None,
        now=1.0,
    )
    assert first["learning_guard_action"] == guard.ACTION_INIT
    assert model.value == 1

    model.value = 2
    second = guard.evaluate(
        stats={"cross_domain_success": 0.4},
        performance_monitor=None,
        predictive_model=model,
        imitation_policy=imitation,
        learning_manager=manager,
        event_bus=None,
        now=2.0,
    )
    assert second["learning_guard_action"] == guard.ACTION_ROLLBACK
    assert model.value == 1
    assert manager.pauses and manager.pauses[-1][1] == "rollback"


def test_learning_safety_guard_pauses_after_patience() -> None:
    guard = LearningSafetyGuard(
        enabled=True,
        patience=2,
        min_delta=0.0,
        rollback_delta=999.0,
        pause_seconds=5.0,
        monitor_window=1,
    )
    model = _DummyModel(1)
    manager = _DummyLearningManager()

    guard.evaluate(
        stats={"cross_domain_success": 0.5},
        performance_monitor=None,
        predictive_model=model,
        imitation_policy=None,
        learning_manager=manager,
        event_bus=None,
        now=1.0,
    )
    guard.evaluate(
        stats={"cross_domain_success": 0.5},
        performance_monitor=None,
        predictive_model=model,
        imitation_policy=None,
        learning_manager=manager,
        event_bus=None,
        now=2.0,
    )
    third = guard.evaluate(
        stats={"cross_domain_success": 0.5},
        performance_monitor=None,
        predictive_model=model,
        imitation_policy=None,
        learning_manager=manager,
        event_bus=None,
        now=3.0,
    )
    assert third["learning_guard_action"] == guard.ACTION_PAUSED
    assert manager.pauses and manager.pauses[-1][1] == "early_stop"
