from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pytest

from modules.learning.continual_learning import ContinualLearningCoordinator, LearningLoopConfig
from modules.learning.experience_hub import ExperienceHub, DemonstrationRecord, EpisodeRecord
from modules.meta_cognition import MetaCognitionController
from modules.autonomy import AutonomousTaskGenerator, MemoryConsolidator


@dataclass
class MetricEvent:
    module: str
    latency: float
    energy: float
    throughput: float
    timestamp: float
    status: str | None = None
    prediction: object | None = None
    actual: object | None = None
    confidence: float | None = None
    stage: str | None = None


class DummyImporter:
    def __init__(self) -> None:
        self.ingested: List[List[Dict[str, object]]] = []

    def ingest_facts(self, facts: Iterable[Dict[str, object]]) -> Dict[str, object]:
        batch = list(facts)
        self.ingested.append(batch)
        return {"count": len(batch)}


class DummyArchitecture:
    def __init__(self, regressions: List[Dict[str, object]]) -> None:
        self.regressions = list(regressions)
        self.calls = 0
        self.coordinator: ContinualLearningCoordinator | None = None

    def run_regression_analysis(self) -> Dict[str, object] | None:
        self.calls += 1
        if self.regressions:
            regression = self.regressions.pop(0)
            if self.coordinator is not None:
                self.coordinator.on_regression_detected(regression)
            return regression
        return None


class DummyCollector:
    def __init__(self, events: List[MetricEvent]) -> None:
        self._events = events

    def events(self) -> List[MetricEvent]:
        return list(self._events)


class DummyCognition:
    def __init__(self) -> None:
        self.observations: List[List[Dict[str, float]] | None] = []

    def observe(self, feedback=None) -> None:  # pragma: no cover - signature compatibility
        if feedback is None:
            self.observations.append(None)
        else:
            self.observations.append(list(feedback))

    @staticmethod
    def feedback_from_event(event: MetricEvent) -> Dict[str, float]:
        correctness = 1.0 if event.status == "success" else 0.0
        return {"correctness": correctness}


class DummyStructuralManager:
    def __init__(self) -> None:
        class _Evolver:
            def fitness_fn(self, arch):
                return float(arch.get("score", 0.0))

        class _Architecture:
            def __init__(self) -> None:
                self.architecture = {"score": 1.0}
                self.evolver = _Evolver()

        self.architecture = _Architecture()
        self.calls: List[Dict[str, object]] = []

    def evolve_structure(self, performance, bottlenecks=None, candidate_modules=None, commit=True):
        self.calls.append(
            {
                "performance": performance,
                "bottlenecks": list(bottlenecks or []),
                "commit": commit,
            }
        )
        self.architecture.architecture["score"] += 0.5


class DummyDiagnoser:
    def __init__(self, issues=None) -> None:
        self.issues = issues or []
        self.calls: int = 0

    def diagnose(self, events=None, aggregate=None):
        self.calls += 1
        return {"issues": list(self.issues)}


class DummyAdjuster:
    def __init__(self) -> None:
        self.calls: int = 0

    def propose(self, issues, current_params=None):
        self.calls += 1
        params = dict(current_params or {})
        params["policy_exploration_rate"] = params.get("policy_exploration_rate", 0.1) + 0.1
        return {"updates": params, "actions": ["bump_exploration"]}


class DummyEngine:
    def __init__(self) -> None:
        class _Cognition:
            def __init__(self) -> None:
                self.architecture = {"policy_exploration_rate": 0.1}
                self.version = 1

            def rollback(self, version):
                self.architecture["rollback_to"] = version
                self.version = version
                return self.architecture

        self.cognition = _Cognition()

    def rollback(self, version):
        return self.cognition.rollback(version)


def test_skill_loop_runs_trainer(tmp_path: Path) -> None:
    hub = ExperienceHub(tmp_path / "episodes")
    results: List[List[Dict[str, object]]] = []

    def trainer(batch: Iterable[Dict[str, object]]) -> None:
        results.append(list(batch))

    coordinator = ContinualLearningCoordinator(
        experience_hub=hub,
        config=LearningLoopConfig(
            skill_interval=0.0,
            knowledge_interval=999.0,
            cognition_interval=999.0,
            architecture_interval=999.0,
            reflection_interval=999.0,
        ),
    )
    coordinator.set_policy_trainer(trainer)
    coordinator.register_episode("task-1", "v1", 1.5, 10, True)

    executed = coordinator.run_maintenance_cycle(force=True)

    assert results and results[0][0]["total_reward"] == pytest.approx(1.5)
    assert executed["skills"] is True


def test_knowledge_loop_ingests_pending_facts(tmp_path: Path) -> None:
    importer = DummyImporter()
    coordinator = ContinualLearningCoordinator(
        experience_hub=ExperienceHub(tmp_path / "episodes"),
        knowledge_importer=importer,
        config=LearningLoopConfig(
            skill_interval=999.0,
            knowledge_interval=0.0,
            cognition_interval=999.0,
            architecture_interval=999.0,
            reflection_interval=999.0,
        ),
    )
    coordinator.register_knowledge_fact({"subject": "dog", "predicate": "isA", "obj": "animal"})

    executed = coordinator.run_maintenance_cycle(force=True)

    assert importer.ingested and importer.ingested[0][0]["subject"] == "dog"
    assert executed["knowledge"] is True


def test_architecture_loop_emits_regression_payload(tmp_path: Path) -> None:
    regressions = [
        {
            "latest_version": 3,
            "reasons": ["resource_decline"],
        }
    ]
    architecture = DummyArchitecture(regressions)
    reflection_payloads: List[Dict[str, object]] = []

    def reflector(payload: Dict[str, object]) -> None:
        reflection_payloads.append(payload)

    coordinator = ContinualLearningCoordinator(
        experience_hub=ExperienceHub(tmp_path / "episodes"),
        architecture=architecture,
        reflection_callback=reflector,
        config=LearningLoopConfig(
            skill_interval=999.0,
            knowledge_interval=999.0,
            cognition_interval=999.0,
            architecture_interval=0.0,
            reflection_interval=0.0,
        ),
    )
    architecture.coordinator = coordinator

    executed = coordinator.run_maintenance_cycle(force=True)

    assert architecture.calls == 1
    assert executed["architecture"] is True
    assert executed["reflection"] is True
    assert reflection_payloads and "regressions" in reflection_payloads[0]
    assert reflection_payloads[0]["regressions"][0]["latest_version"] == 3


def test_cognition_loop_uses_collector_feedback(tmp_path: Path) -> None:
    events = [
        MetricEvent(
            module="planner",
            latency=0.2,
            energy=0.1,
            throughput=5.0,
            timestamp=1.0,
            status="success",
        )
    ]
    cognition = DummyCognition()
    collector = DummyCollector(events)
    coordinator = ContinualLearningCoordinator(
        experience_hub=ExperienceHub(tmp_path / "episodes"),
        cognition=cognition,
        collector=collector,
        config=LearningLoopConfig(
            skill_interval=999.0,
            knowledge_interval=999.0,
            cognition_interval=0.0,
            architecture_interval=999.0,
            reflection_interval=999.0,
        ),
    )

    executed = coordinator.run_maintenance_cycle(force=True)

    assert executed["cognition"] is True
    assert cognition.observations and cognition.observations[0][0]["correctness"] == 1.0


def test_structural_loop_runs_manager(tmp_path: Path) -> None:
    manager = DummyStructuralManager()
    collector = DummyCollector(
        events=[
            MetricEvent(module="x", latency=2.0, energy=0.0, throughput=1.0, timestamp=0.0)
        ]
    )
    coordinator = ContinualLearningCoordinator(
        experience_hub=ExperienceHub(tmp_path / "episodes"),
        collector=collector,
        structural_manager=manager,
        config=LearningLoopConfig(
            skill_interval=999.0,
            knowledge_interval=999.0,
            cognition_interval=999.0,
            architecture_interval=999.0,
            reflection_interval=999.0,
            structural_interval=0.0,
        ),
    )

    executed = coordinator.run_maintenance_cycle(force=True)

    assert executed["structural"] is True
    assert manager.calls, "structural manager should be invoked"
    assert manager.architecture.architecture["score"] > 1.0


def test_diagnosis_loop_runs_and_adjusts(tmp_path: Path) -> None:
    diagnoser = DummyDiagnoser(issues=[{"kind": "high_latency"}])
    adjuster = DummyAdjuster()
    engine = DummyEngine()
    collector = DummyCollector(
        events=[
            MetricEvent(module="plan", latency=0.5, energy=0.0, throughput=1.0, timestamp=0.0)
        ]
    )

    coordinator = ContinualLearningCoordinator(
        experience_hub=ExperienceHub(tmp_path / "episodes"),
        collector=collector,
        performance_diagnoser=diagnoser,
        strategy_adjuster=adjuster,
        evolution_engine=engine,
        config=LearningLoopConfig(
            skill_interval=999.0,
            knowledge_interval=999.0,
            cognition_interval=999.0,
            architecture_interval=999.0,
            reflection_interval=999.0,
            structural_interval=999.0,
            diagnosis_interval=0.0,
        ),
    )

    executed = coordinator.run_maintenance_cycle(force=True)

    assert executed["diagnosis"] is True
    assert diagnoser.calls == 1
    assert adjuster.calls == 1
    assert engine.cognition.architecture["policy_exploration_rate"] > 0.1
    assert engine.cognition.architecture.get("rollback_to") == 0


def test_imitation_loop_consumes_demonstrations(tmp_path: Path) -> None:
    learner_calls: list[int] = []

    class Learner:
        def train(self, demos):
            learner_calls.append(len(list(demos)))

    demo_path = tmp_path / "demo.json"
    demo_path.write_text("{}", encoding="utf-8")

    hub = ExperienceHub(tmp_path / "episodes")
    hub.append_demonstration(
        DemonstrationRecord(
            task_id="demo-task",
            source="human",
            trajectory_path=str(demo_path),
        )
    )

    coordinator = ContinualLearningCoordinator(
        experience_hub=hub,
        imitation_learner=Learner(),
        config=LearningLoopConfig(
            skill_interval=999.0,
            knowledge_interval=999.0,
            cognition_interval=999.0,
            architecture_interval=999.0,
            reflection_interval=999.0,
            structural_interval=999.0,
            diagnosis_interval=999.0,
            imitation_interval=0.0,
            representation_interval=999.0,
        ),
    )

    executed = coordinator.run_maintenance_cycle(force=True)

    assert executed["imitation"] is True
    assert learner_calls and learner_calls[0] == 1


def test_representation_loop_uses_recent_metadata(tmp_path: Path) -> None:
    class RepLearner:
        def __init__(self) -> None:
            self.calls: list[list[dict]] = []

        def train(self, observations):
            self.calls.append(list(observations))

    hub = ExperienceHub(tmp_path / "episodes")
    hub.append(
        EpisodeRecord(
            task_id="t1",
            policy_version="v1",
            total_reward=1.0,
            steps=1,
            success=True,
            metadata={"foo": "bar"},
        )
    )

    coordinator = ContinualLearningCoordinator(
        experience_hub=hub,
        representation_learner=RepLearner(),
        config=LearningLoopConfig(
            skill_interval=999.0,
            knowledge_interval=999.0,
            cognition_interval=999.0,
            architecture_interval=999.0,
            reflection_interval=999.0,
            structural_interval=999.0,
            diagnosis_interval=999.0,
            imitation_interval=999.0,
            representation_interval=0.0,
        ),
    )

    executed = coordinator.run_maintenance_cycle(force=True)

    assert executed["representation"] is True


def test_meta_reflection_loop_marks_signals(tmp_path: Path) -> None:
    controller = MetaCognitionController(failure_threshold=1)
    hub = ExperienceHub(tmp_path / "episodes")
    hub.append(
        EpisodeRecord(
            task_id="t1",
            policy_version="v1",
            total_reward=0.0,
            steps=1,
            success=False,
        )
    )

    coordinator = ContinualLearningCoordinator(
        experience_hub=hub,
        meta_controller=controller,
        config=LearningLoopConfig(
            skill_interval=999.0,
            knowledge_interval=999.0,
            cognition_interval=999.0,
            architecture_interval=999.0,
            reflection_interval=999.0,
            structural_interval=999.0,
            diagnosis_interval=999.0,
            imitation_interval=999.0,
            representation_interval=999.0,
            meta_reflection_interval=0.0,
        ),
    )

    executed = coordinator.run_maintenance_cycle(force=True)

    assert executed["meta_reflection"] is True


def test_register_human_feedback_triggers_knowledge_ingest(tmp_path: Path) -> None:
    importer = DummyImporter()
    coordinator = ContinualLearningCoordinator(
        experience_hub=ExperienceHub(tmp_path / "episodes"),
        knowledge_importer=importer,
        config=LearningLoopConfig(
            skill_interval=999.0,
            knowledge_interval=0.0,
            cognition_interval=999.0,
            architecture_interval=999.0,
            reflection_interval=999.0,
            imitation_interval=999.0,
            meta_reflection_interval=999.0,
        ),
    )
    coordinator.register_human_feedback(
        task_id="t1",
        prompt="What is 2+2?",
        agent_response="5",
        correct_response="4",
        rating=0.0,
    )

    executed = coordinator.run_maintenance_cycle(force=True)

    assert executed["knowledge"] is True
    assert importer.ingested
    assert importer.ingested[0][0]["predicate"] == "answer_is"


def test_task_generation_and_memory_consolidation(tmp_path: Path) -> None:
    generator = AutonomousTaskGenerator()
    consolidator = MemoryConsolidator()
    received: list[list[dict]] = []

    def task_cb(tasks):
        received.append(list(tasks))

    hub = ExperienceHub(tmp_path / "episodes")
    hub.append(
        EpisodeRecord(
            task_id="t1",
            policy_version="v1",
            total_reward=1.0,
            steps=1,
            success=True,
            metadata={"note": "keep"},
        )
    )

    coordinator = ContinualLearningCoordinator(
        experience_hub=hub,
        task_generator=generator,
        memory_consolidator=consolidator,
        task_callback=task_cb,
        config=LearningLoopConfig(
            skill_interval=999.0,
            knowledge_interval=999.0,
            cognition_interval=999.0,
            architecture_interval=999.0,
            reflection_interval=999.0,
            structural_interval=999.0,
            diagnosis_interval=999.0,
            imitation_interval=999.0,
            representation_interval=999.0,
            meta_reflection_interval=999.0,
            task_generation_interval=0.0,
            memory_consolidation_interval=0.0,
        ),
    )

    executed = coordinator.run_maintenance_cycle(force=True)

    assert executed["task_generation"] is True
    assert executed["memory_consolidation"] is True
    assert received and received[0]


def test_attach_memory_consolidator_from_components(tmp_path: Path) -> None:
    class Importer:
        def __init__(self) -> None:
            self.ingested = []

        def ingest_facts(self, facts):
            self.ingested.append(list(facts))

    class VectorStore:
        def __init__(self) -> None:
            self.added = []

        def add_text(self, text, metadata=None, record_id=None):
            self.added.append((text, metadata))
            return "vec-1"

    hub = ExperienceHub(tmp_path / "episodes")
    hub.append(
        EpisodeRecord(
            task_id="t1",
            policy_version="v1",
            total_reward=1.0,
            steps=1,
            success=True,
            metadata={"note": "keep"},
        )
    )

    importer = Importer()
    store = VectorStore()
    coordinator = ContinualLearningCoordinator(
        experience_hub=hub,
        config=LearningLoopConfig(
            skill_interval=999.0,
            knowledge_interval=999.0,
            cognition_interval=999.0,
            architecture_interval=999.0,
            reflection_interval=999.0,
            structural_interval=999.0,
            diagnosis_interval=999.0,
            imitation_interval=999.0,
            representation_interval=999.0,
            meta_reflection_interval=999.0,
            task_generation_interval=999.0,
            memory_consolidation_interval=0.0,
        ),
    )
    coordinator.attach_memory_consolidator_from_components(importer, store)

    executed = coordinator.run_maintenance_cycle(force=True)

    assert executed["memory_consolidation"] is True
    assert importer.ingested  # facts passed to importer
    assert store.added  # summary persisted to vector store
