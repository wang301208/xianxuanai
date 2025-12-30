"""Tests for SelfEvolvingAIArchitecture."""

import json
import os
import random
import sys
import types
from typing import Any

if "psutil" not in sys.modules:
    psutil_stub = types.ModuleType("psutil")

    class _StubProcess:
        def __init__(self, pid: int | None = None) -> None:
            self.pid = pid

        def cpu_percent(self, interval=None):  # pragma: no cover - stub
            return 0.0

        def memory_percent(self):  # pragma: no cover - stub
            return 0.0

        def cpu_times(self):  # pragma: no cover - stub
            return types.SimpleNamespace(user=0.0, system=0.0)

    psutil_stub.Process = _StubProcess
    psutil_stub.NoSuchProcess = RuntimeError
    psutil_stub.AccessDenied = RuntimeError
    sys.modules["psutil"] = psutil_stub

if "fastapi" not in sys.modules:
    fastapi_stub = types.ModuleType("fastapi")

    class _StubFastAPI:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - stub
            pass

        def get(self, *args, **kwargs):  # pragma: no cover - stub
            def decorator(func):
                return func

            return decorator

        def post(self, *args, **kwargs):  # pragma: no cover - stub
            def decorator(func):
                return func

            return decorator

    fastapi_stub.FastAPI = _StubFastAPI
    sys.modules["fastapi"] = fastapi_stub

if "matplotlib" not in sys.modules:
    matplotlib_stub = types.ModuleType("matplotlib")
    pyplot_stub = types.ModuleType("matplotlib.pyplot")

    def _noop(*args, **kwargs):  # pragma: no cover - stub
        return None

    for name in (
        "figure",
        "plot",
        "show",
        "close",
        "subplots",
        "tight_layout",
    ):
        setattr(pyplot_stub, name, _noop)

    matplotlib_stub.pyplot = pyplot_stub
    sys.modules["matplotlib"] = matplotlib_stub
    sys.modules["matplotlib.pyplot"] = pyplot_stub

if "numpy" not in sys.modules:
    numpy_stub = types.ModuleType("numpy")

    def _identity(value=None, *args, **kwargs):  # pragma: no cover - stub
        return value

    numpy_stub.array = _identity
    numpy_stub.mean = lambda *args, **kwargs: 0.0  # pragma: no cover - stub
    numpy_stub.std = lambda *args, **kwargs: 0.0  # pragma: no cover - stub
    numpy_stub.zeros = lambda shape, *args, **kwargs: [0.0] * int(shape[0] if isinstance(shape, (list, tuple)) else shape)  # pragma: no cover
    numpy_stub.ones = lambda shape, *args, **kwargs: [1.0] * int(shape[0] if isinstance(shape, (list, tuple)) else shape)  # pragma: no cover
    numpy_stub.float32 = float
    numpy_stub.ndarray = list
    sys.modules["numpy"] = numpy_stub

if "yaml" not in sys.modules:
    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = _identity  # pragma: no cover - stub
    yaml_stub.safe_dump = lambda *args, **kwargs: ""  # pragma: no cover - stub
    yaml_stub.dump = yaml_stub.safe_dump
    sys.modules["yaml"] = yaml_stub

if "PIL" not in sys.modules:
    pil_stub = types.ModuleType("PIL")
    pil_image_stub = types.ModuleType("PIL.Image")

    class _StubImage:  # pragma: no cover - stub
        def __init__(self, *args, **kwargs) -> None:
            pass

        def save(self, *args, **kwargs) -> None:
            return None

    pil_image_stub.Image = _StubImage
    pil_stub.Image = pil_image_stub
    sys.modules["PIL"] = pil_stub
    sys.modules["PIL.Image"] = pil_image_stub

if "modules.perception.semantic_bridge" not in sys.modules:
    semantic_bridge_stub = types.ModuleType("modules.perception.semantic_bridge")

    class _SemanticBridge:  # pragma: no cover - stub
        def __init__(self, *args, **kwargs) -> None:
            pass

        def encode(self, *args, **kwargs):  # pragma: no cover - stub
            return {}

    semantic_bridge_stub.SemanticBridge = _SemanticBridge
    semantic_bridge_stub.SemanticBridgeOutput = dict
    semantic_bridge_stub.require_default_aligner = lambda: object()
    sys.modules["modules.perception.semantic_bridge"] = semantic_bridge_stub

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.evolution.evolving_cognitive_architecture import (
    EvolvingCognitiveArchitecture,
    GeneticAlgorithm,
    GAConfig,
)
from modules.evolution.evolution_recorder import EvolutionKnowledgeRecorder
from modules.evolution.self_evolving_ai_architecture import SelfEvolvingAIArchitecture
from modules.evolution.self_evolving_cognition import SelfEvolvingCognition
from modules.monitoring.collector import MetricEvent, RealTimeMetricsCollector
from modules.brain.state import BrainRuntimeConfig, CuriosityState


def fitness_fn(arch):
    x = arch["weight"]
    return -(x - 1.0) ** 2


def _basic_setup():
    random.seed(0)
    ga = GeneticAlgorithm(
        fitness_fn, GAConfig(population_size=10, generations=5, mutation_sigma=0.5)
    )
    evolver = EvolvingCognitiveArchitecture(fitness_fn, ga)
    collector = RealTimeMetricsCollector()
    return evolver, collector


# ---------------------------------------------------------------

def test_bottleneck_analysis_to_update():
    evolver, collector = _basic_setup()
    arch = SelfEvolvingAIArchitecture({"weight": 0.0}, evolver, collector)

    collector._events.extend(
        [
            MetricEvent("A", latency=2.0, energy=0.0, throughput=1.0, timestamp=0.0),
            MetricEvent("A", latency=3.0, energy=0.0, throughput=1.0, timestamp=1.0),
            MetricEvent("B", latency=1.0, energy=0.0, throughput=1.0, timestamp=2.0),
        ]
    )

    bottlenecks = arch.analyze_performance_bottlenecks()
    assert bottlenecks[0][0] == "A"

    candidates = arch.generate_architecture_mutations()
    best = arch.evolutionary_selection(candidates)
    assert arch.architecture == arch._normalise_architecture(best)
    assert len(arch.history) > 1
    arch.rollback(0)
    assert arch.architecture == arch.history[0].architecture


# ---------------------------------------------------------------

def test_history_shared_with_cognition():
    evolver, collector = _basic_setup()
    cognition = SelfEvolvingCognition({"weight": 0.0}, evolver, collector)
    arch = SelfEvolvingAIArchitecture(
        cognition.architecture, evolver, collector, cognition
    )

    assert arch.history is cognition.history

    collector._events.append(
        MetricEvent("mod", latency=0.1, energy=0.0, throughput=1.0, timestamp=0.0)
    )
    bottlenecks = arch.analyze_performance_bottlenecks()
    candidates = arch.generate_architecture_mutations()
    arch.evolutionary_selection(candidates)
    assert cognition.architecture == arch.architecture
    arch.rollback(0)
    assert arch.architecture == cognition.architecture == arch.history[0].architecture


def test_update_persists_extended_genome():
    evolver, collector = _basic_setup()

    class StubMemoryManager:
        def __init__(self) -> None:
            self._short_term_limit = 25
            self._working_limit = 50
            self._summary_batch_size = 5
            self._summary_rate_limit = 900.0

    memory_manager = StubMemoryManager()
    curiosity = CuriosityState()
    brain_config = BrainRuntimeConfig()

    class StubPolicy:
        def __init__(self) -> None:
            self.learning_rate = 0.08
            self.exploration = 0.12

    class StubLearner:
        def __init__(self) -> None:
            self.learning_rate = 0.05

    class StubReflection:
        def __init__(self) -> None:
            self.reflection_interval_hours = 16.0
            self.calls: list[float] = []

        def set_reflection_interval(self, hours: float) -> None:
            self.reflection_interval_hours = hours
            self.calls.append(hours)

    class StubCognitiveModule:
        def __init__(self, policy: StubPolicy) -> None:
            self.policy = policy

    policy = StubPolicy()
    learner = StubLearner()
    reflection = StubReflection()
    cognitive_module = StubCognitiveModule(policy)
    arch = SelfEvolvingAIArchitecture(
        {"weight": 0.0},
        evolver,
        collector,
        memory_manager=memory_manager,
        curiosity_state=curiosity,
        brain_config=brain_config,
        policy_module=cognitive_module,
        learning_modules=[learner],
        reflection_controller=reflection,
    )

    defaults = arch._defaults
    assert defaults["policy_learning_rate"] == policy.learning_rate
    assert defaults["policy_exploration_rate"] == policy.exploration
    assert defaults["memory_summary_batch_size"] == float(memory_manager._summary_batch_size)
    assert defaults["reflection_interval_hours"] == reflection.reflection_interval_hours
    assert defaults["module_self_learning_flag"] == 1.0
    assert defaults["module_metrics_flag"] == 1.0
    assert defaults["module_curiosity_feedback_flag"] == 1.0

    new_arch = arch.architecture.copy()
    new_arch.update(
        {
            "memory_short_term_limit": 40.0,
            "memory_working_limit": 80.0,
            "curiosity_drive_floor": 0.7,
            "curiosity_novelty_preference": 0.8,
            "curiosity_fatigue_ceiling": 0.2,
            "planner_reinforcement_flag": 1.0,
            "planner_structured_flag": 0.0,
            "policy_learning_rate": 0.2,
            "policy_exploration_rate": 0.3,
            "memory_summary_batch_size": 9.0,
            "memory_summary_rate_limit": 1_200.0,
            "reflection_interval_hours": 12.0,
            "module_self_learning_flag": 0.0,
            "module_curiosity_feedback_flag": 1.0,
            "module_metrics_flag": 0.0,
        }
    )
    arch.update_architecture(new_arch, performance=1.0)

    assert memory_manager._short_term_limit == 40
    assert memory_manager._working_limit == 80
    assert memory_manager._summary_batch_size == 9
    assert memory_manager._summary_rate_limit == 1200.0
    assert curiosity.drive == 0.7
    assert curiosity.novelty_preference == 0.8
    assert curiosity.fatigue == 0.2
    assert brain_config.prefer_reinforcement_planner is True
    assert brain_config.prefer_structured_planner is False
    assert brain_config.enable_plan_logging is False
    assert brain_config.enable_self_learning is False
    assert brain_config.enable_curiosity_feedback is True
    assert brain_config.metrics_enabled is False
    assert policy.learning_rate == 0.2
    assert policy.exploration == 0.3
    assert learner.learning_rate == 0.2
    assert reflection.reflection_interval_hours == 12.0
    assert reflection.calls[-1] == 12.0


def test_behavioural_fitness_bonus():
    evolver, collector = _basic_setup()

    class DummyStorage:
        def __init__(self, rate: float) -> None:
            self._rate = rate

        def success_rate(self) -> float:
            return self._rate

    class DummyMonitor:
        def __init__(self) -> None:
            self.storage = DummyStorage(0.8)

        def log_resource_usage(self, *args, **kwargs) -> None:  # pragma: no cover - stub
            pass

        def log_task_completion(self, *args, **kwargs) -> None:  # pragma: no cover - stub
            pass

    collector._monitor = DummyMonitor()
    collector._events.extend(
        [
            MetricEvent("policy", latency=0.1, energy=0.05, throughput=5.0, timestamp=0.0),
            MetricEvent("policy", latency=0.2, energy=0.05, throughput=4.0, timestamp=0.1),
        ]
    )

    arch = SelfEvolvingAIArchitecture({"weight": 0.0}, evolver, collector)

    base = arch._base_fitness_fn(arch.architecture)
    combined = arch.evolver.fitness_fn(arch.architecture)
    assert combined > base


def test_behaviour_metrics_from_collector_outcomes():
    evolver, collector = _basic_setup()
    arch = SelfEvolvingAIArchitecture({"weight": 0.0}, evolver, collector)

    collector._events.clear()
    collector._events.extend(
        [
            MetricEvent(
                "module",
                latency=0.2,
                energy=0.1,
                throughput=2.0,
                timestamp=0.0,
                status="success",
            ),
            MetricEvent(
                "module",
                latency=0.4,
                energy=0.2,
                throughput=1.0,
                timestamp=0.1,
                status="failure",
            ),
        ]
    )

    metrics = arch._collect_behaviour_metrics()
    assert metrics["success_rate"] == 0.5


def test_normalise_architecture_clamps_new_params():
    evolver, collector = _basic_setup()
    arch = SelfEvolvingAIArchitecture({"weight": 0.0}, evolver, collector)

    normalised = arch._normalise_architecture(
        {
            "memory_short_term_limit": -10.0,
            "memory_working_limit": -20.0,
            "policy_learning_rate": -1.0,
            "policy_exploration_rate": 2.0,
            "cognitive_policy_variant": 9.0,
            "planner_min_steps": 0.0,
            "policy_replay_buffer_size": 0.0,
            "policy_replay_batch_size": 0.0,
            "policy_replay_iterations": 0.0,
            "policy_hidden_dim": 0.0,
            "policy_num_layers": 0.0,
            "memory_summary_batch_size": 0.2,
            "memory_summary_rate_limit": 10.0,
            "reflection_interval_hours": 0.01,
            "module_self_learning_flag": 0.49,
            "module_curiosity_feedback_flag": 0.51,
            "module_metrics_flag": 2.0,
        }
    )

    assert normalised["memory_short_term_limit"] >= 1.0
    assert normalised["memory_working_limit"] >= normalised["memory_short_term_limit"]
    assert normalised["policy_learning_rate"] >= 1e-5
    assert normalised["policy_exploration_rate"] <= 1.0
    assert normalised["policy_exploration_rate"] >= 0.0
    assert normalised["cognitive_policy_variant"] == 2.0
    assert normalised["planner_min_steps"] >= 1.0
    assert normalised["policy_replay_buffer_size"] >= 32.0
    assert normalised["policy_replay_batch_size"] >= 1.0
    assert normalised["policy_replay_iterations"] >= 1.0
    assert normalised["policy_hidden_dim"] >= 8.0
    assert normalised["policy_num_layers"] >= 1.0
    assert normalised["memory_summary_batch_size"] == 1.0
    assert normalised["memory_summary_rate_limit"] >= 60.0
    assert normalised["reflection_interval_hours"] >= 0.25
    assert normalised["module_self_learning_flag"] == 0.0
    assert normalised["module_curiosity_feedback_flag"] == 1.0
    assert normalised["module_metrics_flag"] == 1.0


def test_policy_variant_switches_policy_and_replay_params():
    evolver, collector = _basic_setup()

    from modules.brain.whole_brain_policy import ProductionCognitivePolicy, ReinforcementCognitivePolicy

    class PolicyContainer:
        def __init__(self) -> None:
            self.policy = ProductionCognitivePolicy()

        def set_policy(self, policy: Any) -> None:
            self.policy = policy

    container = PolicyContainer()
    arch = SelfEvolvingAIArchitecture({"weight": 0.0}, evolver, collector, policy_module=container)

    updated = arch.architecture.copy()
    updated.update(
        {
            "cognitive_policy_variant": 1.0,
            "planner_min_steps": 7.0,
            "policy_learning_rate": 0.2,
            "policy_exploration_rate": 0.3,
            "policy_replay_buffer_size": 64.0,
            "policy_replay_batch_size": 4.0,
            "policy_replay_iterations": 2.0,
        }
    )
    arch.update_architecture(updated, performance=1.0)

    assert isinstance(container.policy, ReinforcementCognitivePolicy)
    assert container.policy.learning_rate == 0.2
    assert container.policy.exploration == 0.3
    assert container.policy.planner.min_steps == 7
    assert container.policy._experience_buffer.maxlen == 64
    assert container.policy._replay_batch_size == 4
    assert container.policy._replay_iterations == 2


def test_regression_detection_triggers_learning_program(tmp_path):
    evolver = EvolvingCognitiveArchitecture(lambda arch: arch.get("weight", 0.0))
    recorder = EvolutionKnowledgeRecorder(
        log_path=tmp_path / "evolution.jsonl", enable_graph=False
    )

    class MemoryManager:
        def __init__(self) -> None:
            self.requests: list[dict[str, Any]] = []

        def schedule_review(self, **kwargs: Any) -> None:
            self.requests.append(kwargs)

    class PolicyModule:
        def __init__(self) -> None:
            self.training: list[dict[str, Any]] = []

        def schedule_additional_training(self, **kwargs: Any) -> None:
            self.training.append(kwargs)

    class ReflectionController:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def schedule_reflection(self, **kwargs: Any) -> None:
            self.calls.append(kwargs)

    class PracticeModule:
        def __init__(self) -> None:
            self.sessions: list[dict[str, Any]] = []

        def schedule_practice(self, **kwargs: Any) -> None:
            self.sessions.append(kwargs)

    memory_manager = MemoryManager()
    policy_module = PolicyModule()
    reflection = ReflectionController()
    practice = PracticeModule()

    architecture = SelfEvolvingAIArchitecture(
        {"weight": 1.0},
        evolver,
        cognition=None,
        collector=None,
        recorder=recorder,
        memory_manager=memory_manager,
        policy_module=policy_module,
        learning_modules=[practice],
        reflection_controller=reflection,
    )

    architecture.update_architecture(
        {"weight": 0.95},
        performance=0.95,
        metrics={"resource_score": 0.95, "success_rate": 0.55},
    )
    architecture.update_architecture(
        {"weight": 0.9},
        performance=0.9,
        metrics={"resource_score": 0.9, "success_rate": 0.4},
    )
    architecture.update_architecture(
        {"weight": 0.85},
        performance=0.6,
        metrics={"resource_score": 0.6, "success_rate": 0.35},
    )

    assert memory_manager.requests
    assert policy_module.training
    assert reflection.calls
    assert practice.sessions

    with recorder.log_path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    payload = json.loads(lines[-1])
    regression = payload["annotations"]["regression"]
    assert regression["reasons"]
    assert regression["interventions"]
