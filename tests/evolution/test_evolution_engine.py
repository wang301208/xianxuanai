import json
import sys
import types
from pathlib import Path


def ensure_stub(name: str, **attrs):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


ensure_stub("autogpts")
ensure_stub("third_party.autogpt")
ensure_stub("third_party.autogpt.autogpt")
ensure_stub("third_party.autogpt.autogpt.core")
ensure_stub("third_party.autogpt.autogpt.core.errors", AutoGPTError=RuntimeError)
ensure_stub("third_party.autogpt.autogpt.core.logging", handle_exception=lambda exc: None)


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


class _StubEvolutionKnowledgeRecorder:
    def __init__(self, *args, **kwargs):
        self.calls = []
        log_path = kwargs.get("log_path")
        if log_path is None and args:
            log_path = args[0]
        self.log_path = Path(log_path) if log_path is not None else Path("results") / "evolution_stub.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if not args:
            return
        record = args[0]
        previous_architecture = kwargs.get("previous_architecture")
        annotations = kwargs.get("annotations")

        try:
            payload = {
                "version": getattr(record, "version", None),
                "performance": getattr(record, "performance", None),
                "architecture": getattr(record, "architecture", None),
                "metrics": getattr(record, "metrics", None),
            }
            if previous_architecture is not None:
                payload["delta"] = {"previous_architecture": previous_architecture}
            if annotations is not None:
                payload["annotations"] = annotations
            with self.log_path.open("a", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, default=str)
                handle.write("\n")
        except Exception:
            return


ensure_stub(
    "modules.evolution.evolution_recorder",
    EvolutionKnowledgeRecorder=_StubEvolutionKnowledgeRecorder,
)

import pytest

from modules.evolution.cognitive_benchmark import CognitiveBenchmarkResult
from modules.evolution.evolution_engine import (
    EvolutionEngine,
    SpecialistModule,
    SpecialistModuleRegistry,
    TaskContext,
)
from modules.evolution import (
    StructuralEvolutionManager,
    StructuralGenome,
    ModuleGene,
    SelfEvolvingAIArchitecture,
    EvolvingCognitiveArchitecture,
)
from modules.monitoring.collector import MetricEvent
from modules.monitoring import PerformanceDiagnoser
from modules.evolution import StrategyAdjuster


class DummyGA:
    """Deterministic GA stub to keep evolution predictable in tests."""

    def __init__(self, fitness_fn, best_architecture):
        self._fitness_fn = fitness_fn
        self.best_architecture = best_architecture
        self.calls = []

    def evolve(self, architecture):
        self.calls.append(architecture.copy())
        score = self._fitness_fn(self.best_architecture)
        return self.best_architecture.copy(), score, [(self.best_architecture.copy(), score)]


def make_metric(module="planner", throughput=120.0, latency=0.2, energy=0.1):
    return MetricEvent(
        module=module,
        latency=latency,
        energy=energy,
        throughput=throughput,
        timestamp=1.0,
        stage=module,
    )


def test_specialist_registry_prefers_highest_average_score():
    def _solver(arch, task):
        return arch

    experienced = SpecialistModule(
        name="experienced",
        capabilities={"analysis"},
        solver=_solver,
        priority=0.1,
    )
    experienced.record_performance(0.9)
    experienced.record_performance(0.7)  # average = 0.8

    flashy = SpecialistModule(
        name="flashy",
        capabilities={"analysis"},
        solver=_solver,
        priority=1.0,
    )
    flashy.record_performance(0.5)  # average = 0.5

    registry = SpecialistModuleRegistry([experienced, flashy])
    task = TaskContext(name="Analysis task", required_capabilities=["Analysis"])

    assert registry.select_best(task) is experienced


def test_evolution_engine_prefers_specialist_when_performance_higher():
    initial_architecture = {"accuracy": 0.5}

    def fitness_fn(arch):
        return arch.get("accuracy", 0.0)

    ga = DummyGA(fitness_fn, {"accuracy": 0.6})
    specialist_architecture = {"accuracy": 0.95}

    def specialist_solver(architecture, task):
        return {**architecture, **specialist_architecture}

    specialist = SpecialistModule(
        name="accuracy_booster",
        capabilities={"analysis"},
        solver=specialist_solver,
        priority=0.5,
    )

    engine = EvolutionEngine(
        initial_architecture=initial_architecture,
        fitness_fn=fitness_fn,
        ga=ga,
        specialist_modules=[specialist],
    )

    metrics = [make_metric(module="analysis", throughput=100.0, latency=0.1, energy=0.05)]
    task = TaskContext(name="Planning", required_capabilities=["ANALYSIS"])

    result = engine.run_evolution_cycle(metrics=metrics, benchmarks=None, task=task)

    assert result == specialist_architecture
    assert engine.cognition.architecture == specialist_architecture
    assert engine.cognition.history[-1].metrics["source"] == "specialist"
    assert engine.cognition.history[-1].metrics["specialist_module"] == "accuracy_booster"
    assert engine.cognition.history[-1].performance == pytest.approx(0.95)
    assert specialist.usage_count == 1
    assert ga.calls, "GA should still be invoked to generate baseline candidate"


def test_evolution_cycle_handles_metrics_and_benchmarks_together():
    initial_architecture = {"accuracy": 0.5}

    def fitness_fn(arch):
        return arch.get("accuracy", 0.0)

    ga = DummyGA(fitness_fn, {"accuracy": 0.55})
    engine = EvolutionEngine(
        initial_architecture=initial_architecture,
        fitness_fn=fitness_fn,
        ga=ga,
        score_weights={"resource": 0.7, "cognitive": 0.3},
    )

    metrics = [make_metric(throughput=80.0, latency=0.2, energy=0.1)]
    benchmarks = [
        CognitiveBenchmarkResult(task_id="task-1", success=True, latency=10.0, reward=0.4, quality=0.9),
        CognitiveBenchmarkResult(task_id="task-2", success=False, latency=12.0, reward=0.1, quality=0.5),
    ]

    result = engine.run_evolution_cycle(metrics=metrics, benchmarks=benchmarks, task=None)

    assert result == {"accuracy": 0.55}
    latest_record = engine.cognition.history[-1]
    assert latest_record.metrics["combined_performance"] != 0.0
    assert "success_rate" in latest_record.metrics
    assert latest_record.metrics["source"] == "genetic"


def test_run_evolution_cycle_no_inputs_returns_current_architecture():
    initial_architecture = {"accuracy": 0.42}

    def fitness_fn(arch):
        return arch.get("accuracy", 0.0)

    ga = DummyGA(fitness_fn, {"accuracy": 0.8})
    engine = EvolutionEngine(
        initial_architecture=initial_architecture,
        fitness_fn=fitness_fn,
        ga=ga,
    )

    result = engine.run_evolution_cycle(metrics=[], benchmarks=None, task=None)

    assert result == initial_architecture
    assert len(engine.cognition.history) == 1  # only the initial record is present


def test_evolution_engine_uses_structural_candidate():
    base_architecture = {"accuracy": 0.2, "module_extra_active": 0.0}

    def fitness_fn(arch):
        return arch.get("accuracy", 0.0) + 0.5 * arch.get("module_extra_active", 0.0)

    ga = DummyGA(fitness_fn, {"accuracy": 0.3, "module_extra_active": 0.0})

    struct_evolver = EvolvingCognitiveArchitecture(fitness_fn)
    structural_ai = SelfEvolvingAIArchitecture(base_architecture.copy(), struct_evolver)
    genome = StructuralGenome(
        modules=[ModuleGene("base"), ModuleGene("extra", enabled=False)],
        edges=[],
    )
    structural_manager = StructuralEvolutionManager(
        architecture=structural_ai,
        genome=genome,
        exploration_budget=2,
    )

    engine = EvolutionEngine(
        initial_architecture=base_architecture,
        fitness_fn=fitness_fn,
        ga=ga,
        structural_manager=structural_manager,
    )

    metrics = [make_metric(module="base", throughput=50.0, latency=0.3, energy=0.1)]

    result = engine.run_evolution_cycle(metrics=metrics, benchmarks=None, task=None)

    assert result.get("module_extra_active") == 1.0
    assert engine.cognition.history[-1].metrics["source"] == "structural"
    assert structural_manager.module_gates["extra"] == 1.0


def test_structural_interval_respected():
    base_architecture = {"accuracy": 0.2, "module_extra_active": 0.0}

    def fitness_fn(arch):
        return arch.get("accuracy", 0.0) + 0.5 * arch.get("module_extra_active", 0.0)

    ga = DummyGA(fitness_fn, {"accuracy": 0.25, "module_extra_active": 0.0})

    struct_evolver = EvolvingCognitiveArchitecture(fitness_fn)
    structural_ai = SelfEvolvingAIArchitecture(base_architecture.copy(), struct_evolver)
    genome = StructuralGenome(
        modules=[ModuleGene("base"), ModuleGene("extra", enabled=False)],
        edges=[],
    )
    structural_manager = StructuralEvolutionManager(
        architecture=structural_ai,
        genome=genome,
        exploration_budget=2,
    )

    engine = EvolutionEngine(
        initial_architecture=base_architecture,
        fitness_fn=fitness_fn,
        ga=ga,
        structural_manager=structural_manager,
        structural_interval=2,
    )

    metrics = [make_metric(module="base", throughput=50.0, latency=0.3, energy=0.1)]

    first = engine.run_evolution_cycle(metrics=metrics, benchmarks=None, task=None)
    assert first.get("module_extra_active", 0.0) == 0.0
    assert engine.cognition.history[-1].metrics["source"] != "structural"

    second = engine.run_evolution_cycle(metrics=metrics, benchmarks=None, task=None)
    assert second.get("module_extra_active") == 1.0
    assert engine.cognition.history[-1].metrics["source"] == "structural"


def test_diagnoser_and_strategy_adjuster_influence_architecture():
    initial_architecture = {"policy_exploration_rate": 0.1}

    def fitness_fn(arch):
        return arch.get("policy_exploration_rate", 0.0)

    ga = DummyGA(fitness_fn, {"policy_exploration_rate": 0.1})
    diagnoser = PerformanceDiagnoser(max_latency_s=0.05, min_success_rate=0.9)
    adjuster = StrategyAdjuster(exploration_step=0.2)

    engine = EvolutionEngine(
        initial_architecture=initial_architecture,
        fitness_fn=fitness_fn,
        ga=ga,
        performance_diagnoser=diagnoser,
        strategy_adjuster=adjuster,
    )

    metrics = [
        MetricEvent(
            module="planner",
            latency=0.2,
            energy=0.0,
            throughput=1.0,
            timestamp=1.0,
            status="failure",
        )
    ]

    result = engine.run_evolution_cycle(metrics=metrics, benchmarks=None, task=None)

    assert result["policy_exploration_rate"] > 0.1
    metrics_summary = engine.cognition.history[-1].metrics
    assert "diagnostic_issues" in metrics_summary
    assert metrics_summary["diagnostic_penalty"] > 0.0
    assert "strategy_actions" in metrics_summary


def test_elite_guard_prevents_regression():
    initial_architecture = {"score": 1.0}

    def fitness_fn(arch):
        return arch.get("score", 0.0)

    class DescendingGA:
        def __init__(self):
            self.calls = 0

        def evolve(self, arch):
            self.calls += 1
            worse = dict(arch)
            worse["score"] = arch.get("score", 0.0) - 0.5
            return worse, worse["score"], [(worse, worse["score"])]

    ga = DescendingGA()
    engine = EvolutionEngine(
        initial_architecture=initial_architecture,
        fitness_fn=fitness_fn,
        ga=ga,
        enforce_elite=True,
    )

    metrics = [make_metric(module="plan", throughput=1.0, latency=0.1, energy=0.0)]
    result = engine.run_evolution_cycle(metrics=metrics, benchmarks=None, task=None)

    assert result["score"] == 1.0  # elite preserved
    assert engine.cognition.history[-1].metrics["source"] == "elite_guard"
