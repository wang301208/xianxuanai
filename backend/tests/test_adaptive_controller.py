import importlib.util
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

BACKEND_EXECUTION = Path(ROOT, "backend", "execution")


def _load(name: str):
    module_path = BACKEND_EXECUTION / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"backend.execution.{name}", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module  # type: ignore[attr-defined]
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


    module = _load("adaptive_controller")
    AdaptiveResourceController = module.AdaptiveResourceController  # type: ignore[attr-defined]
MetaParameterSpec = module.MetaParameterSpec  # type: ignore[attr-defined]
EpsilonGreedyMetaPolicy = module.EpsilonGreedyMetaPolicy  # type: ignore[attr-defined]
HybridArchitectureManager = module.HybridArchitectureManager  # type: ignore[attr-defined]
GAConfig = getattr(module, "GAConfig")  # type: ignore[attr-defined]


class StubBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def publish(self, topic: str, payload: dict) -> None:
        self.events.append((topic, payload))


class StubMemoryRouter:
    def __init__(self) -> None:
        self.calls = 0

    def shrink(self, **_: object) -> int:
        self.calls += 1
        return 3


class StubLongTermMemory:
    def __init__(self) -> None:
        self.calls = 0

    def compress(self) -> None:
        self.calls += 1


class StubConfig:
    def __init__(self) -> None:
        self.big_brain = True
        self.goal_frequency = 1
        self.parallel_branches = 1.0


def test_memory_pressure_triggers_compaction(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = StubBus()
    router = StubMemoryRouter()
    long_term = StubLongTermMemory()
    config = StubConfig()

    controller = AdaptiveResourceController(
        config=config,
        event_bus=bus,
        memory_router=router,
        long_term_memory=long_term,
        memory_threshold=10.0,
        memory_cooldown=0.0,
    )

    controller.update(avg_cpu=0.0, avg_memory=50.0, backlog=0)

    assert router.calls == 1
    assert long_term.calls == 1
    assert bus.events and bus.events[-1][0] == "resource.adaptation.memory"


def test_cpu_pressure_switches_modes() -> None:
    bus = StubBus()
    config = StubConfig()
    controller = AdaptiveResourceController(
        config=config,
        event_bus=bus,
        memory_router=None,
        long_term_memory=None,
        cpu_high_threshold=10.0,
        cpu_recover_threshold=5.0,
        mode_cooldown=0.0,
    )

    controller.update(avg_cpu=50.0, avg_memory=0.0, backlog=2)
    assert config.big_brain is False
    assert bus.events and bus.events[-1][0] == "resource.adaptation.mode"
    assert bus.events[-1][1]["mode"] == "low"

    controller.update(avg_cpu=0.0, avg_memory=0.0, backlog=0)
    assert config.big_brain is True
    assert bus.events[-1][1]["mode"] == "high"


def test_meta_policy_updates_bandit() -> None:
    bus = StubBus()
    config = StubConfig()
    policy = EpsilonGreedyMetaPolicy(
        [MetaParameterSpec("goal_frequency", [1, 2], epsilon=0.0, alpha=0.5)]
    )
    controller = AdaptiveResourceController(
        config=config,
        event_bus=bus,
        memory_router=None,
        long_term_memory=None,
        meta_policy=policy,
        reward_fn=lambda metrics: metrics.get("reward"),
        rng_seed=7,
    )

    controller.update(avg_cpu=0.0, avg_memory=0.0, backlog=0, metrics={"reward": 0.1})
    first_choice = config.goal_frequency
    controller.update(avg_cpu=0.0, avg_memory=0.0, backlog=0, metrics={"reward": 0.9})

    spec = policy._specs["goal_frequency"]  # type: ignore[attr-defined]
    index = spec.values.index(first_choice)
    counts = policy._counts["goal_frequency"]  # type: ignore[attr-defined]
    assert counts[index] >= 1
    assert config.goal_frequency in spec.values


def test_architecture_manager_triggers_evolution() -> None:
    applied: list[dict] = []

    def evaluator(arch: Mapping[str, float], samples: Sequence[Mapping[str, Any]]) -> float:
        base = sum(float(sample.get("reward", 0.0) or 0.0) for sample in samples) / max(len(samples), 1)
        return base + float(arch.get("parallel_branches", 0.0)) * 0.1

    manager = HybridArchitectureManager(
        initial_architecture={"parallel_branches": 1.0},
        evaluator=evaluator,
        apply_callback=lambda arch: applied.append(arch),
        ga_config=GAConfig(population_size=4, generations=1, mutation_sigma=0.4),
        history=deque(maxlen=3),
        cooldown_steps=1,
        min_improvement=-0.01,
        seed=5,
    )

    bus = StubBus()
    config = StubConfig()
    controller = AdaptiveResourceController(
        config=config,
        event_bus=bus,
        memory_router=None,
        long_term_memory=None,
        architecture_manager=manager,
        reward_fn=lambda metrics: metrics.get("reward"),
        rng_seed=3,
    )

    for reward in (0.2, 0.3, 0.5, 0.7):
        controller.update(avg_cpu=0.0, avg_memory=0.0, backlog=0, metrics={"reward": reward})

    assert applied, "Expected architecture callback to run"
    assert any(topic == "resource.adaptation.architecture" for topic, _ in bus.events)


def test_architecture_hotloader_applies_runtime_values() -> None:
    class Memory:
        def __init__(self) -> None:
            self._short_term_limit = 4
            self._working_limit = 8
            self._summary_batch_size = 2
            self._summary_rate_limit = 120.0

    class BrainCfg:
        def __init__(self) -> None:
            self.prefer_structured_planner = True
            self.prefer_reinforcement_planner = False
            self.enable_self_learning = True
            self.enable_curiosity_feedback = True
            self.metrics_enabled = True

    class Policy:
        def __init__(self) -> None:
            self.learning_rate = 0.1
            self.exploration = 0.2

    runtime_config = StubConfig()
    memory = Memory()
    brain = BrainCfg()
    policy = Policy()

    hotloader = module.ArchitectureHotloader(  # type: ignore[attr-defined]
        runtime_config=runtime_config,
        brain_config=brain,
        memory_manager=memory,
        policy_module=policy,
    )

    baseline = hotloader.derive_baseline()
    tweaked = dict(baseline)
    tweaked["memory_short_term_limit"] = baseline["memory_short_term_limit"] + 3
    tweaked["planner_structured_flag"] = 0.0
    tweaked["big_brain_flag"] = 0.0
    tweaked["policy_learning_rate"] = baseline["policy_learning_rate"] * 0.5
    hotloader.apply(tweaked)

    assert runtime_config.big_brain is False
    assert brain.prefer_structured_planner is False
    assert memory._short_term_limit == int(round(tweaked["memory_short_term_limit"]))
    assert policy.learning_rate == float(tweaked["policy_learning_rate"])
    assert hotloader.last_applied is not None


def test_architecture_manager_from_runtime_hotloads() -> None:
    bus = StubBus()
    runtime_config = StubConfig()
    policy = type("P", (), {"learning_rate": 0.1, "exploration": 0.1})()
    manager = module.HybridArchitectureManager.from_runtime(  # type: ignore[attr-defined]
        runtime_config=runtime_config,
        policy_module=policy,
        history=deque(maxlen=1),
        ga_config=GAConfig(population_size=3, generations=1, mutation_sigma=0.2),
        cooldown_steps=0,
        min_improvement=-0.5,
        seed=11,
    )

    controller = AdaptiveResourceController(
        config=runtime_config,
        event_bus=bus,
        memory_router=None,
        long_term_memory=None,
        architecture_manager=manager,
        reward_fn=lambda metrics: metrics.get("reward", 0.0),
        rng_seed=9,
    )

    controller.update(avg_cpu=0.0, avg_memory=0.0, backlog=0, metrics={"reward": 0.5, "success_rate": 0.8})

    assert manager.hotloader and manager.hotloader.last_applied is not None
    assert any(topic == "resource.adaptation.architecture" for topic, _ in bus.events)


def test_architecture_manager_pso_refine(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force PSO to return a known better position.
    import numpy as np  # type: ignore

    def stub_pso(obj, bounds, **kwargs):  # type: ignore
        pos = np.array([(low + high) / 2 for low, high in bounds], dtype=float)
        return type("Res", (), {"position": pos, "value": obj(pos)})()

    monkeypatch.setattr(module, "pso", stub_pso)  # type: ignore[attr-defined]

    def evaluator(arch: Mapping[str, float], samples: Sequence[Mapping[str, Any]]) -> float:
        # Maximise negative L2 distance to origin => best at zeros.
        return -sum(float(arch.get(k, 0.0)) ** 2 for k in ("x", "y"))

    manager = HybridArchitectureManager(
        initial_architecture={"x": 1.0, "y": 1.0},
        evaluator=evaluator,
        apply_callback=None,
        ga_config=GAConfig(population_size=3, generations=1, mutation_sigma=0.1),
        history=deque(maxlen=1),
        cooldown_steps=0,
        min_improvement=-0.5,
        pso_bounds={"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
        pso_config={"num_particles": 4, "max_iter": 5},
        seed=3,
    )

    event = manager.observe({"reward": 0.0})
    assert event is not None
    assert abs(manager.current_architecture()["x"]) < 0.75


def test_module_adapter_toggle() -> None:
    class _Module:
        def __init__(self) -> None:
            self.enabled = True

    mod = _Module()
    calls: list[str] = []

    def enable() -> None:
        mod.enabled = True
        calls.append("enable")

    def disable() -> None:
        mod.enabled = False
        calls.append("disable")

    adapter = module.ModuleAdapter(  # type: ignore[attr-defined]
        name="dummy",
        enable=enable,
        disable=disable,
        enabled_probe=lambda: mod.enabled,
    )

    hotloader = module.ArchitectureHotloader(  # type: ignore[attr-defined]
        runtime_config=StubConfig(),
        module_adapters=[adapter],
    )
    base = hotloader.derive_baseline()
    assert base["module_dummy_flag"] == 1.0

    tweaked = dict(base)
    tweaked["module_dummy_flag"] = 0.0
    hotloader.apply(tweaked)
    assert mod.enabled is False
    assert "disable" in calls


def test_monitor_snapshot_best_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyMonitor:
        def __init__(self) -> None:
            self.logged = []

        def log_inference(self, value: float) -> None:
            self.logged.append(("inference", value))

        def log_training(self, value: float) -> None:
            self.logged.append(("training", value))

        def log_resource(self) -> None:
            self.logged.append(("resource", 1.0))

        def log_snapshot(self, metrics: dict) -> None:
            self.logged.append(("snapshot", metrics))

    controller = AdaptiveResourceController(
        config=StubConfig(),
        event_bus=None,
        memory_router=None,
        long_term_memory=None,
        monitor=DummyMonitor(),
    )

    controller.update(avg_cpu=10.0, avg_memory=20.0, backlog=1, metrics={"reward": 0.2, "success_rate": 0.8})
    assert any(kind == "snapshot" for kind, _ in controller._monitor.logged)  # type: ignore[attr-defined]
    assert any(kind == "resource" for kind, _ in controller._monitor.logged)  # type: ignore[attr-defined]


def test_internal_feedback_evaluator_flags_low_confidence() -> None:
    controller = AdaptiveResourceController(
        config=StubConfig(),
        event_bus=None,
        memory_router=None,
        long_term_memory=None,
    )
    controller.record_extra_metrics(
        {"perception_confidence_avg": 0.2, "decision_success_rate": 0.4, "memory_hit_rate": 0.0}
    )
    controller.update(avg_cpu=0.0, avg_memory=0.0, backlog=0)
    assert controller._feedback_log  # type: ignore[attr-defined]


def test_feedback_triggers_meta_adjustment(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def meta_adjust(payload: dict) -> None:
        calls.append(payload)

    controller = AdaptiveResourceController(
        config=StubConfig(),
        event_bus=None,
        memory_router=None,
        long_term_memory=None,
        meta_adjustment_provider=meta_adjust,
    )
    controller.record_extra_metrics({"decision_success_rate": 0.2, "decision_reward_avg": -0.1})
    controller.update(avg_cpu=0.0, avg_memory=0.0, backlog=0)
    assert calls
    assert any(call.get("reason") == "low_reward_avg" for call in calls)


def test_self_improvement_manager_integration() -> None:
    controller = AdaptiveResourceController(
        config=StubConfig(),
        event_bus=None,
        memory_router=None,
        long_term_memory=None,
    )
    controller.record_extra_metrics(
        {
            "decision_success_rate": 0.6,
            "decision_reward_avg": 0.05,
            "perception_prediction_error": 0.2,
            "memory_hit_rate": 0.1,
        }
    )
    controller.update(avg_cpu=0.0, avg_memory=0.0, backlog=0)
    # Goals should tighten after observation
    sim = controller._self_improvement  # type: ignore[attr-defined]
    assert sim is not None
    assert "decision_success_rate" in sim.goals
    assert "decision_reward_avg" in sim.goals


def test_meta_learning_tuner_emits_adjustments(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def meta_adjust(payload: dict) -> None:
        calls.append(payload)

    controller = AdaptiveResourceController(
        config=StubConfig(),
        event_bus=None,
        memory_router=None,
        long_term_memory=None,
        meta_adjustment_provider=meta_adjust,
    )
    for _ in range(3):
        controller.record_extra_metrics({"decision_success_rate": 0.3, "decision_reward_avg": -0.2})
        controller.update(avg_cpu=0.0, avg_memory=0.0, backlog=0)
    assert any(call.get("reason") == "meta_learning" for call in calls)

