import sys
import types
from importlib import util
from pathlib import Path
from typing import Any, Dict, List

import pytest


ROOT = Path(__file__).resolve().parents[2]
backend_pkg = types.ModuleType("backend")
backend_pkg.__path__ = [str(ROOT / "backend")]
sys.modules.setdefault("backend", backend_pkg)
execution_pkg = types.ModuleType("backend.execution")
execution_pkg.__path__ = [str(ROOT / "backend" / "execution")]
sys.modules.setdefault("backend.execution", execution_pkg)

MODULE_PATH = ROOT / "backend" / "execution" / "self_improvement.py"
spec = util.spec_from_file_location("backend.execution.self_improvement", MODULE_PATH)
module = util.module_from_spec(spec)
assert spec and spec.loader
sys.modules.setdefault("backend.execution.self_improvement", module)
spec.loader.exec_module(module)

SelfImprovementManager = module.SelfImprovementManager


class _NullExperienceStore:
    def record(self, record: Dict[str, Any]) -> None:
        return

    def rank_kinds(self, *, metric: str, kinds: List[str]) -> List[str]:
        del metric
        return list(kinds)

    def stats_for(self, *, metric: str, kinds: List[str]) -> Dict[str, Any]:
        del metric, kinds
        return {}


def test_self_improvement_prefers_experience_ranked_action() -> None:
    class _PreferBigBrainStore(_NullExperienceStore):
        def rank_kinds(self, *, metric: str, kinds: List[str]) -> List[str]:
            if metric != "decision_success_rate":
                return list(kinds)
            ordered = ["decision_big_brain", "decision_exploration_boost"]
            return [k for k in ordered if k in kinds] + [k for k in kinds if k not in ordered]

    sim = SelfImprovementManager(
        enabled=True,
        default_patience=2,
        default_cooldown_secs=0.0,
        eval_window=2,
        min_improvement=0.1,
        experience_store=_PreferBigBrainStore(),
    )
    sim.ensure_goal("decision_success_rate", 0.8, direction="increase", patience=2, cooldown_secs=0.0)

    sim.observe_metrics({"decision_success_rate": 0.2})
    sim.observe_metrics({"decision_success_rate": 0.2})

    class _Cfg:
        inference_uniform_mix = 0.0

    class _Policy:
        config = _Cfg()

    class _Runtime:
        big_brain = False

    policy = _Policy()
    runtime = _Runtime()

    sim.run_next(imitation_policy=policy, runtime_config=runtime)
    assert runtime.big_brain is True
    assert policy.config.inference_uniform_mix == 0.0


def test_self_improvement_rolls_back_failed_exploration_boost() -> None:
    sim = SelfImprovementManager(
        enabled=True,
        default_patience=2,
        default_cooldown_secs=0.0,
        eval_window=2,
        min_improvement=0.1,
        experience_store=_NullExperienceStore(),
    )
    sim.ensure_goal("decision_success_rate", 0.8, direction="increase", patience=2, cooldown_secs=0.0)

    # Trigger sustained under-performance.
    sim.observe_metrics({"decision_success_rate": 0.2})
    sim.observe_metrics({"decision_success_rate": 0.2})

    class _Cfg:
        inference_uniform_mix = 0.0

    class _Policy:
        config = _Cfg()

    policy = _Policy()
    sim.run_next(imitation_policy=policy)
    assert policy.config.inference_uniform_mix > 0.0

    # No meaningful improvement -> evaluation should rollback.
    sim.observe_metrics({"decision_success_rate": 0.22})
    sim.observe_metrics({"decision_success_rate": 0.25})
    sim.run_next(imitation_policy=policy)

    assert policy.config.inference_uniform_mix == 0.0


def test_self_improvement_schedules_perception_retrain() -> None:
    sim = SelfImprovementManager(
        enabled=True,
        default_patience=2,
        default_cooldown_secs=0.0,
        experience_store=_NullExperienceStore(),
    )
    sim.ensure_goal("perception_confidence_avg", 0.6, direction="increase", patience=2, cooldown_secs=0.0)

    sim.observe_metrics({"perception_confidence_avg": 0.1})
    sim.observe_metrics({"perception_confidence_avg": 0.1})

    calls: List[Dict[str, Any]] = []
    sim.run_next(retrain_callback=lambda payload: calls.append(dict(payload)))

    assert calls
    assert calls[-1].get("module") == "perception"


def test_self_improvement_knowledge_update_ingests_docs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / "docs.md").write_text("domainX: explanation of the missing concept", encoding="utf-8")

    class _Pipeline:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def process_task_event(self, event: dict) -> None:
            self.events.append(dict(event))

    class _Memory:
        def __init__(self) -> None:
            self.observations: list[dict] = []

        def add_observation(self, text: str, *, source: str, metadata: dict) -> None:
            self.observations.append({"text": text, "source": source, "metadata": dict(metadata)})

    pipeline = _Pipeline()
    memory = _Memory()

    sim = SelfImprovementManager(
        enabled=True,
        default_patience=1,
        default_cooldown_secs=0.0,
        experience_store=_NullExperienceStore(),
    )
    sim.ensure_goal("knowledge_gaps", 0.0, direction="decrease", patience=1, cooldown_secs=0.0)

    sim.observe_metrics({"knowledge_gaps": 2.0, "knowledge_gap_domains": ["domainX"]})
    sim.run_next(knowledge_pipeline=pipeline, memory_router=memory, now=1.0)

    assert pipeline.events
    assert pipeline.events[-1].get("knowledge_statements")
    assert memory.observations


def test_self_improvement_applies_automl_suggestion_and_rolls_back() -> None:
    class _Bus:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict]] = []

        def publish(self, topic: str, event: dict) -> None:
            self.events.append((topic, dict(event)))

    bus = _Bus()
    sim = SelfImprovementManager(
        enabled=True,
        eval_window=2,
        min_improvement=0.1,
        default_patience=999,
        default_cooldown_secs=0.0,
        experience_store=_NullExperienceStore(),
    )
    sim.ensure_goal("decision_success_rate", 0.8, direction="increase", patience=999, cooldown_secs=0.0)
    sim.observe_metrics({"decision_success_rate": 0.2})

    class _Cfg:
        lr = 0.05
        inference_uniform_mix = 0.0

    class _Policy:
        config = _Cfg()

    class _Runtime:
        big_brain = True

    policy = _Policy()
    runtime = _Runtime()

    assert sim.enqueue_automl_suggestion(
        {
            "time": 1.0,
            "suggestion_id": "s-1",
            "metric": "decision_success_rate",
            "params": {"imitation.lr": 0.2, "runtime.big_brain": False},
        }
    )

    sim.run_next(event_bus=bus, imitation_policy=policy, runtime_config=runtime)
    assert policy.config.lr == pytest.approx(0.2)
    assert runtime.big_brain is False

    sim.observe_metrics({"decision_success_rate": 0.21})
    sim.observe_metrics({"decision_success_rate": 0.25})
    sim.run_next(event_bus=bus, imitation_policy=policy, runtime_config=runtime)

    assert policy.config.lr == pytest.approx(0.05)
    assert runtime.big_brain is True

    feedback = [event for topic, event in bus.events if topic == "automl.feedback"]
    assert feedback
    assert feedback[-1]["suggestion_id"] == "s-1"
    assert feedback[-1]["metric"] == "decision_success_rate"
    assert isinstance(feedback[-1]["objective_value"], float)


def test_self_improvement_requests_automl_when_stagnant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SELF_IMPROVEMENT_AUTOML_REQUEST_ENABLED", "1")
    monkeypatch.setenv("SELF_IMPROVEMENT_AUTOML_COOLDOWN_SECS", "0")
    monkeypatch.setenv("SELF_IMPROVEMENT_AUTOML_STAGNATION_WINDOW", "3")
    monkeypatch.setenv("SELF_IMPROVEMENT_AUTOML_STAGNATION_MIN_DELTA", "0.05")
    monkeypatch.setenv("SELF_IMPROVEMENT_AUTOML_TARGET_ONLY", "1")

    class _Bus:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict]] = []

        def publish(self, topic: str, event: dict) -> None:
            self.events.append((topic, dict(event)))

    bus = _Bus()
    sim = SelfImprovementManager(
        enabled=True,
        default_patience=999,
        default_cooldown_secs=0.0,
        experience_store=_NullExperienceStore(),
    )
    sim.ensure_goal("decision_success_rate", 0.8, direction="increase", patience=999, cooldown_secs=0.0)

    sim.observe_metrics({"decision_success_rate": 0.2})
    sim.observe_metrics({"decision_success_rate": 0.21})
    sim.observe_metrics({"decision_success_rate": 0.22})

    stats = sim.run_next(event_bus=bus, now=10.0)
    assert stats.get("self_improvement_automl_requested") == 1.0
    requests = [event for topic, event in bus.events if topic == "automl.request"]
    assert requests
    assert requests[-1]["metric"] == "decision_success_rate"
