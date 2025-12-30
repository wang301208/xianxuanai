import sys
import types


def ensure_stub(name: str, **attrs):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


class _FakePerformanceMonitor:
    def log_resource_usage(self, *args, **kwargs):
        return None

    def log_prediction(self, *args, **kwargs):
        return None

    def log_task_completion(self, *args, **kwargs):
        return None

    def log_task_result(self, *args, **kwargs):
        return None


ensure_stub("backend")
ensure_stub("backend.monitoring", PerformanceMonitor=_FakePerformanceMonitor)
sys.modules["backend"].__path__ = []
ensure_stub("backend.knowledge")
ensure_stub("backend.knowledge.registry", get_graph_store_instance=lambda: None)
sys.modules["backend.knowledge"].__path__ = []


class _FakeProcess:
    def __init__(self, pid=None):
        self.pid = pid

    def cpu_times(self):
        return types.SimpleNamespace(user=0.0, system=0.0)

    def memory_percent(self):
        return 0.0

    def cpu_percent(self, interval=None):
        return 0.0


ensure_stub(
    "psutil",
    Process=_FakeProcess,
    NoSuchProcess=RuntimeError,
    AccessDenied=RuntimeError,
)


from modules.evolution.evolving_cognitive_architecture import EvolvingCognitiveArchitecture  # noqa: E402
from modules.evolution.self_evolving_ai_architecture import SelfEvolvingAIArchitecture  # noqa: E402


def test_safety_gate_rejects_regression():
    def fitness_fn(arch):
        return float(arch.get("score", 0.0))

    evolver = EvolvingCognitiveArchitecture(fitness_fn)
    arch = SelfEvolvingAIArchitecture(
        {"score": 1.0},
        evolver,
        safety_config={
            "enabled": True,
            "history_window": 1,
            "min_relative_to_mean": 0.9,
        },
    )

    assert arch.architecture["score"] == 1.0
    accepted = arch.update_architecture({"score": 0.5}, performance=0.5)
    assert accepted is False
    assert arch.architecture["score"] == 1.0


def test_manual_review_queue_and_approval():
    def fitness_fn(arch):
        return float(arch.get("score", 0.0))

    evolver = EvolvingCognitiveArchitecture(fitness_fn)
    arch = SelfEvolvingAIArchitecture(
        {"score": 1.0},
        evolver,
        safety_config={
            "enabled": True,
            "manual_review_enabled": True,
            "manual_review_delta_l1": 0.1,
        },
    )

    accepted = arch.update_architecture({"score": 2.0}, performance=2.0)
    assert accepted is False
    assert arch.architecture["score"] == 1.0

    pending = arch.list_pending_architecture_updates(limit=10)
    assert len(pending) == 1
    pending_id = pending[0].id

    approved = arch.approve_pending_architecture_update(pending_id, run_sandbox=False)
    assert approved is True
    assert arch.architecture["score"] == 2.0


def test_sandbox_runner_blocks_update():
    def fitness_fn(arch):
        return float(arch.get("score", 0.0))

    evolver = EvolvingCognitiveArchitecture(fitness_fn)

    def failing_sandbox():
        return False

    arch = SelfEvolvingAIArchitecture(
        {"score": 1.0},
        evolver,
        safety_config={
            "enabled": True,
            "sandbox_enabled": True,
            "sandbox_runner": failing_sandbox,
        },
    )

    accepted = arch.update_architecture({"score": 2.0}, performance=2.0)
    assert accepted is False
    assert arch.architecture["score"] == 1.0
