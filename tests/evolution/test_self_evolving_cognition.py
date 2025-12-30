"""Tests for the SelfEvolvingCognition module."""

import importlib
import importlib.util
import os
import random
import sys
import types
from pathlib import Path
from types import ModuleType, SimpleNamespace

sys.path.insert(0, os.path.abspath(os.getcwd()))

if "psutil" not in sys.modules:
    process_stub = lambda *_, **__: types.SimpleNamespace(
        cpu_percent=lambda interval=None: 0.0,
        memory_percent=lambda: 0.0,
    )
    sys.modules["psutil"] = types.SimpleNamespace(
        Process=lambda *args, **kwargs: process_stub(),
        NoSuchProcess=Exception,
        AccessDenied=Exception,
    )

if "backend.monitoring" not in sys.modules:
    monitor_stub = ModuleType("backend.monitoring")

    class _PerformanceMonitorStub:
        def log_resource_usage(self, *_args, **_kwargs) -> None:
            pass

        def log_task_completion(self, *_args, **_kwargs) -> None:
            pass

    monitor_stub.PerformanceMonitor = _PerformanceMonitorStub
    backend_pkg = importlib.import_module("backend")
    setattr(backend_pkg, "monitoring", monitor_stub)
    sys.modules["backend.monitoring"] = monitor_stub

if "fastapi" not in sys.modules:
    class _FastAPIStub:
        def __init__(self, *_, **__):
            pass

        def get(self, *_args, **_kwargs):
            def decorator(func):
                return func

            return decorator

        def post(self, *_args, **_kwargs):
            def decorator(func):
                return func

            return decorator

    sys.modules["fastapi"] = types.SimpleNamespace(FastAPI=_FastAPIStub)

if "matplotlib" not in sys.modules:
    pyplot_stub = types.SimpleNamespace(
        figure=lambda *args, **kwargs: None,
        plot=lambda *args, **kwargs: None,
        tight_layout=lambda *args, **kwargs: None,
        savefig=lambda *args, **kwargs: None,
        close=lambda *args, **kwargs: None,
    )
    matplotlib_stub = ModuleType("matplotlib")
    matplotlib_stub.pyplot = pyplot_stub
    sys.modules["matplotlib"] = matplotlib_stub
    sys.modules["matplotlib.pyplot"] = pyplot_stub

if "numpy" not in sys.modules:
    numpy_stub = ModuleType("numpy")
    numpy_stub.array = lambda *args, **kwargs: args[0] if args else []
    numpy_stub.ndarray = object
    numpy_stub.mean = lambda data, *_, **__: sum(data) / len(data) if data else 0.0
    numpy_stub.std = lambda data, *_, **__: 0.0
    sys.modules["numpy"] = numpy_stub

ROOT = Path(__file__).resolve().parents[2]

if "modules" not in sys.modules:
    modules_pkg = ModuleType("modules")
    modules_pkg.__path__ = [str(ROOT / "modules")]
    sys.modules["modules"] = modules_pkg

if "modules.evolution" not in sys.modules:
    evolution_pkg = ModuleType("modules.evolution")
    evolution_pkg.__path__ = [str(ROOT / "modules" / "evolution")]
    sys.modules["modules.evolution"] = evolution_pkg

if "modules.monitoring" not in sys.modules:
    monitoring_pkg = ModuleType("modules.monitoring")
    monitoring_pkg.__path__ = [str(ROOT / "modules" / "monitoring")]
    sys.modules["modules.monitoring"] = monitoring_pkg


def _load_module(name: str, relative_path: str):
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Cannot load module {name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evo_module = _load_module("modules.evolution.evolving_cognitive_architecture", "modules/evolution/evolving_cognitive_architecture.py")
cog_module = _load_module("modules.evolution.self_evolving_cognition", "modules/evolution/self_evolving_cognition.py")
monitor_module = _load_module("modules.monitoring.collector", "modules/monitoring/collector.py")

EvolvingCognitiveArchitecture = evo_module.EvolvingCognitiveArchitecture
GAConfig = evo_module.GAConfig
GeneticAlgorithm = evo_module.GeneticAlgorithm
SelfEvolvingCognition = cog_module.SelfEvolvingCognition
MetricEvent = monitor_module.MetricEvent
RealTimeMetricsCollector = monitor_module.RealTimeMetricsCollector


def fitness_fn(arch):
    x = arch["weight"]
    return -(x - 1.0) ** 2


def test_long_term_evolution_improves_performance():
    random.seed(0)
    ga = GeneticAlgorithm(fitness_fn, GAConfig(population_size=10, generations=5, mutation_sigma=0.5))
    evolver = EvolvingCognitiveArchitecture(fitness_fn, ga)
    collector = RealTimeMetricsCollector()
    cognition = SelfEvolvingCognition({"weight": 0.0}, evolver, collector)

    for step in range(5):
        perf = fitness_fn(cognition.architecture)
        event = MetricEvent(
            module="evolve", latency=0.0, energy=0.0, throughput=perf, timestamp=float(step)
        )
        collector._events.append(event)
        cognition.observe()

    performances = [rec.performance for rec in cognition.history]
    assert performances[-1] > performances[0]
    cognition.rollback(0)
    assert cognition.architecture == cognition.history[0].architecture
    diff = cognition.compare(0, len(cognition.history) - 1)
    assert diff["performance_diff"] == performances[-1] - performances[0]


class DummyEvolver:
    def __init__(self) -> None:
        self.last_performance: float | None = None

    def evolve_architecture(self, architecture, performance):
        self.last_performance = performance
        new_arch = architecture.copy()
        new_arch["weight"] = new_arch.get("weight", 0.0) + performance
        return new_arch

    def fitness_fn(self, architecture):
        return architecture.get("weight", 0.0)


def test_reflection_feedback_shapes_scoring_and_history():
    evolver = DummyEvolver()
    cognition = SelfEvolvingCognition(
        {"weight": 0.0},
        evolver,
        collector=None,
        scoring_weights={"throughput": 1.0, "latency": 0.0, "energy": 0.0, "confidence": 2.0, "correctness": 3.0},
    )

    event = MetricEvent(
        module="unit",
        latency=0.0,
        energy=0.0,
        throughput=1.0,
        timestamp=0.0,
    )
    feedback = SimpleNamespace(confidence=0.4, success=0.0)

    cognition._process_event(event, evaluation=feedback)

    expected_score = 1.0 + 2.0 * 0.4 + 3.0 * 0.0
    assert evolver.last_performance == expected_score

    last_record = cognition.history[-1]
    assert last_record.confidence == 0.4
    assert last_record.correctness == 0.0
    assert last_record.metrics["confidence"] == 0.4
    assert last_record.metrics["correctness"] == 0.0


def test_observe_ingests_feedback_and_adapter_outputs_metadata():
    evolver = DummyEvolver()
    collector = RealTimeMetricsCollector()
    cognition = SelfEvolvingCognition(
        {"weight": 0.0},
        evolver,
        collector=collector,
        scoring_weights={"throughput": 1.0, "latency": 0.0, "energy": 0.0},
    )

    event_one = MetricEvent(
        module="unit",
        latency=0.0,
        energy=0.0,
        throughput=1.0,
        timestamp=1.0,
    )
    collector._events.append(event_one)

    cognition.observe([{"confidence": 0.8, "correctness": 0.6}])

    first_record = cognition.history[-1]
    assert first_record.confidence == 0.8
    assert first_record.correctness == 0.6
    assert first_record.metrics["confidence"] == 0.8
    assert first_record.metrics["correctness"] == 0.6

    event_two = MetricEvent(
        module="unit",
        latency=0.0,
        energy=0.0,
        throughput=2.0,
        timestamp=2.0,
        status="failure",
        confidence=0.25,
        prediction="foo",
        actual="bar",
    )
    collector._events.append(event_two)

    cognition.observe(SelfEvolvingCognition.feedback_from_event)

    second_record = cognition.history[-1]
    assert second_record.confidence == 0.25
    assert second_record.correctness == 0.0
    assert second_record.metrics["confidence"] == 0.25
    assert second_record.metrics["correctness"] == 0.0
