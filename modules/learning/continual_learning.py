"""Coordinators for continual, self-supervised learning loops."""

from __future__ import annotations

"""Coordinators for continual, self-supervised learning loops."""

import threading
import time
import json
from dataclasses import dataclass, field, is_dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, TYPE_CHECKING

from .experience_hub import DemonstrationRecord, EpisodeRecord, ExperienceHub

try:  # Optional knowledge stack dependency
    from modules.knowledge import KnowledgeFact, RuntimeKnowledgeImporter
except Exception:  # pragma: no cover - fallback when knowledge stack absent
    @dataclass
    class KnowledgeFact:  # type: ignore[override]
        subject: str
        predicate: str
        obj: str
        metadata: Dict[str, Any] = field(default_factory=dict)

    class RuntimeKnowledgeImporter:  # type: ignore[override]
        def ingest_facts(self, _facts: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
            raise RuntimeError("Knowledge importer unavailable")

if TYPE_CHECKING:  # pragma: no cover - typing only
    from modules.evolution.self_evolving_cognition import SelfEvolvingCognition
    from modules.evolution.self_evolving_ai_architecture import SelfEvolvingAIArchitecture
    from modules.monitoring.collector import RealTimeMetricsCollector
    from modules.evolution.structural_evolution import StructuralEvolutionManager
    from modules.monitoring.performance_diagnoser import PerformanceDiagnoser
    from modules.evolution.strategy_adjuster import StrategyAdjuster
    from modules.evolution.evolution_engine import EvolutionEngine
    from modules.autonomy import MemoryConsolidator


LoopCallback = Callable[["ContinualLearningCoordinator"], bool | None]


@dataclass
class LearningLoopConfig:
    """Configuration for each continual learning loop."""

    skill_interval: float = 900.0
    knowledge_interval: float = 1200.0
    cognition_interval: float = 600.0
    architecture_interval: float = 1800.0
    reflection_interval: float = 1800.0
    background_interval: float = 60.0
    structural_interval: float = 3600.0
    diagnosis_interval: float = 600.0
    imitation_interval: float = 900.0
    representation_interval: float = 3600.0
    meta_reflection_interval: float = 7200.0
    task_generation_interval: float = 900.0
    memory_consolidation_interval: float = 43200.0


@dataclass
class _LoopState:
    callback: LoopCallback
    interval: float
    last_run: float = field(default_factory=lambda: 0.0)
    enabled: bool = True


class ContinualLearningCoordinator:
    """Manage multi-layer learning loops across the agent stack."""

    def __init__(
        self,
        *,
        experience_hub: ExperienceHub | None = None,
        experience_root: Path | str | None = None,
        knowledge_importer: RuntimeKnowledgeImporter | None = None,
        cognition: "SelfEvolvingCognition" | None = None,
        architecture: "SelfEvolvingAIArchitecture" | None = None,
        structural_manager: "StructuralEvolutionManager" | None = None,
        performance_diagnoser: "PerformanceDiagnoser" | None = None,
        strategy_adjuster: "StrategyAdjuster" | None = None,
        evolution_engine: "EvolutionEngine" | None = None,
        imitation_learner: Any | None = None,
        representation_learner: Any | None = None,
        meta_controller: Any | None = None,
        task_generator: Any | None = None,
        memory_consolidator: Any | None = None,
        task_callback: Callable[[List[Dict[str, Any]]], Any] | None = None,
        task_scheduler: Any | None = None,
        collector: "RealTimeMetricsCollector" | None = None,
        policy_trainer: Callable[[Iterable[Dict[str, Any]]], Any] | None = None,
        reflection_callback: Callable[[Dict[str, Any]], Any] | None = None,
        config: LearningLoopConfig | None = None,
        executor: Callable[[LoopCallback, "ContinualLearningCoordinator"], Any] | None = None,
    ) -> None:
        self.config = config or LearningLoopConfig()
        if experience_hub is None:
            root = Path(experience_root) if experience_root is not None else Path("data/experience")
            experience_hub = ExperienceHub(Path(root))
        self.experience_hub = experience_hub
        self._knowledge_importer = knowledge_importer
        self._cognition = cognition
        self._architecture = architecture
        self._structural_manager = structural_manager
        self._performance_diagnoser = performance_diagnoser
        self._strategy_adjuster = strategy_adjuster
        self._evolution_engine = evolution_engine
        self._imitation_learner = imitation_learner
        self._representation_learner = representation_learner
        self._meta_controller = meta_controller
        self._task_generator = task_generator
        self._memory_consolidator = memory_consolidator
        self._task_callback = task_callback
        self._task_scheduler = task_scheduler
        self._collector = collector
        self._policy_trainer = policy_trainer
        self._reflection_callback = reflection_callback
        self._executor = executor

        self._lock = threading.Lock()
        self._loops: Dict[str, _LoopState] = {}
        self._dirty_loops: set[str] = set()
        self._pending_facts: List[Dict[str, Any]] = []
        self._pending_regressions: List[Dict[str, Any]] = []
        self._worker_thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None

        self._register_builtin_loops()

    # ------------------------------------------------------------------
    def _register_builtin_loops(self) -> None:
        if self.experience_hub is not None:
            self.register_loop(
                "skills",
                self._skill_loop,
                self.config.skill_interval,
            )
        if self._knowledge_importer is not None:
            self.register_loop(
                "knowledge",
                self._knowledge_loop,
                self.config.knowledge_interval,
            )
        if self._cognition is not None:
            self.register_loop(
                "cognition",
                self._cognition_loop,
                self.config.cognition_interval,
            )
        if self._architecture is not None:
            self.register_loop(
                "architecture",
                self._architecture_loop,
                self.config.architecture_interval,
            )
        if self._reflection_callback is not None:
            self.register_loop(
                "reflection",
                self._reflection_loop,
                self.config.reflection_interval,
            )
        if self._structural_manager is not None:
            self.register_loop(
                "structural",
                self._structural_loop,
                self.config.structural_interval,
            )
        if self._performance_diagnoser is not None:
            self.register_loop(
                "diagnosis",
                self._diagnosis_loop,
                self.config.diagnosis_interval,
            )
        if self._imitation_learner is not None:
            self.register_loop(
                "imitation",
                self._imitation_loop,
                self.config.imitation_interval,
            )
        if self._representation_learner is not None:
            self.register_loop(
                "representation",
                self._representation_loop,
                self.config.representation_interval,
            )
        if self._meta_controller is not None:
            self.register_loop(
                "meta_reflection",
                self._meta_reflection_loop,
                self.config.meta_reflection_interval,
            )
        if self._task_generator is not None:
            self.register_loop(
                "task_generation",
                self._task_generation_loop,
                self.config.task_generation_interval,
            )
        if self._memory_consolidator is not None:
            self.register_loop(
                "memory_consolidation",
                self._memory_consolidation_loop,
                self.config.memory_consolidation_interval,
            )

    # ------------------------------------------------------------------
    def register_loop(
        self,
        name: str,
        callback: LoopCallback,
        interval: float,
        *,
        enabled: bool = True,
        immediate: bool = False,
    ) -> None:
        """Register or update a named learning loop."""

        state = _LoopState(callback=callback, interval=float(interval), enabled=enabled)
        with self._lock:
            self._loops[name] = state
            if immediate:
                self._dirty_loops.add(name)

    # ------------------------------------------------------------------
    def set_loop_enabled(self, name: str, enabled: bool) -> None:
        """Enable or disable a registered loop by name."""

        with self._lock:
            state = self._loops.get(name)
            if state is None:
                raise KeyError(f"Loop '{name}' is not registered")
            state.enabled = bool(enabled)
            if enabled:
                self._dirty_loops.add(name)

    # ------------------------------------------------------------------
    def attach_cognition(self, cognition: "SelfEvolvingCognition") -> None:
        self._cognition = cognition
        self.register_loop(
            "cognition",
            self._cognition_loop,
            self.config.cognition_interval,
            enabled=True,
            immediate=True,
        )

    # ------------------------------------------------------------------
    def attach_architecture(self, architecture: "SelfEvolvingAIArchitecture") -> None:
        self._architecture = architecture
        self.register_loop(
            "architecture",
            self._architecture_loop,
            self.config.architecture_interval,
            enabled=True,
            immediate=True,
        )

    # ------------------------------------------------------------------
    def attach_knowledge_importer(self, importer: RuntimeKnowledgeImporter) -> None:
        """Attach a knowledge importer and enable the knowledge loop."""

        self._knowledge_importer = importer
        self.register_loop(
            "knowledge",
            self._knowledge_loop,
            self.config.knowledge_interval,
            enabled=True,
            immediate=False,
        )

    # ------------------------------------------------------------------
    def attach_structural_manager(self, manager: "StructuralEvolutionManager") -> None:
        self._structural_manager = manager
        self.register_loop(
            "structural",
            self._structural_loop,
            self.config.structural_interval,
            enabled=True,
            immediate=False,
        )

    # ------------------------------------------------------------------
    def attach_diagnoser(
        self,
        diagnoser: "PerformanceDiagnoser",
        strategy_adjuster: "StrategyAdjuster" | None = None,
        evolution_engine: "EvolutionEngine" | None = None,
    ) -> None:
        self._performance_diagnoser = diagnoser
        if strategy_adjuster is not None:
            self._strategy_adjuster = strategy_adjuster
        if evolution_engine is not None:
            self._evolution_engine = evolution_engine
        self.register_loop(
            "diagnosis",
            self._diagnosis_loop,
            self.config.diagnosis_interval,
            enabled=True,
            immediate=False,
        )

    # ------------------------------------------------------------------
    def attach_meta_controller(self, controller: Any) -> None:
        self._meta_controller = controller
        self.register_loop(
            "meta_reflection",
            self._meta_reflection_loop,
            self.config.meta_reflection_interval,
            enabled=True,
            immediate=False,
        )

    # ------------------------------------------------------------------
    def attach_task_generator(self, generator: Any) -> None:
        self._task_generator = generator
        self.register_loop(
            "task_generation",
            self._task_generation_loop,
            self.config.task_generation_interval,
            enabled=True,
            immediate=False,
        )

    # ------------------------------------------------------------------
    def set_task_callback(self, callback: Callable[[List[Dict[str, Any]]], Any]) -> None:
        """Set a handler to receive generated tasks."""

        self._task_callback = callback

    # ------------------------------------------------------------------
    def attach_task_scheduler(self, scheduler: Any) -> None:
        self._task_scheduler = scheduler

    # ------------------------------------------------------------------
    def attach_memory_consolidator(self, consolidator: Any) -> None:
        self._memory_consolidator = consolidator
        self.register_loop(
            "memory_consolidation",
            self._memory_consolidation_loop,
            self.config.memory_consolidation_interval,
            enabled=True,
            immediate=False,
        )

    # ------------------------------------------------------------------
    def attach_memory_consolidator_from_components(
        self,
        knowledge_importer: Any,
        vector_store: Any | None = None,
    ) -> None:
        """Convenience: build and attach MemoryConsolidator with real stores."""

        try:
            from modules.autonomy import MemoryConsolidator
        except Exception:  # pragma: no cover - defensive import
            MemoryConsolidator = None  # type: ignore[assignment]
        if MemoryConsolidator is None:
            raise RuntimeError("MemoryConsolidator unavailable")
        consolidator = MemoryConsolidator(knowledge_importer=knowledge_importer, vector_store=vector_store)
        self.attach_memory_consolidator(consolidator)

    # ------------------------------------------------------------------
    def set_collector(self, collector: "RealTimeMetricsCollector") -> None:
        self._collector = collector

    # ------------------------------------------------------------------
    def set_policy_trainer(
        self, trainer: Callable[[Iterable[Dict[str, Any]]], Any]
    ) -> None:
        self._policy_trainer = trainer
        if trainer is not None:
            self._dirty_loops.add("skills")

    # ------------------------------------------------------------------
    def set_reflection_callback(
        self, callback: Callable[[Dict[str, Any]], Any]
    ) -> None:
        self._reflection_callback = callback
        self.register_loop(
            "reflection",
            self._reflection_loop,
            self.config.reflection_interval,
            enabled=True,
            immediate=True,
        )

    # ------------------------------------------------------------------
    def register_episode(
        self,
        task_id: str,
        policy_version: str,
        total_reward: float,
        steps: int,
        success: bool,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        trajectory_path: str | None = None,
    ) -> EpisodeRecord:
        """Persist a new learning episode and mark the skills loop dirty."""

        record = EpisodeRecord(
            task_id=task_id,
            policy_version=policy_version,
            total_reward=float(total_reward),
            steps=int(steps),
            success=bool(success),
            metadata=metadata or {},
            trajectory_path=trajectory_path,
        )
        self.experience_hub.append(record)
        self._dirty_loops.add("skills")
        return record

    # ------------------------------------------------------------------
    def register_human_feedback(
        self,
        *,
        task_id: str,
        prompt: str,
        agent_response: str,
        correct_response: str | None = None,
        rating: float | None = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DemonstrationRecord:
        """Persist a human-supervised feedback record and trigger learning loops."""

        payload = {
            "task_id": str(task_id or "task"),
            "prompt": str(prompt or ""),
            "agent_response": str(agent_response or ""),
            "correct_response": str(correct_response or "") if correct_response is not None else None,
            "rating": float(rating) if rating is not None else None,
            "metadata": dict(metadata or {}),
        }
        root = getattr(self.experience_hub, "root", None) if self.experience_hub is not None else None
        trajectory_path = None
        if root is not None:
            try:
                path = Path(root) / "human_feedback"
                path.mkdir(parents=True, exist_ok=True)
                target = path / f"feedback_{int(time.time())}.json"
                target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                trajectory_path = str(target)
            except Exception:
                trajectory_path = None

        demo = DemonstrationRecord(
            task_id=str(task_id or "task"),
            source="human",
            trajectory_path=trajectory_path,
            metadata=dict(metadata or {}),
        )
        if self.experience_hub is not None:
            self.experience_hub.append_demonstration(demo)
        self._dirty_loops.add("imitation")

        # Optional: convert corrections into a simple knowledge fact.
        if correct_response is not None and str(correct_response).strip():
            self.register_knowledge_fact(
                {
                    "subject": str(prompt or "").strip()[:256] or str(task_id or "task"),
                    "predicate": "answer_is",
                    "obj": str(correct_response).strip()[:512],
                    "metadata": {"source": "human_feedback", "rating": rating},
                }
            )
            self._dirty_loops.add("knowledge")

        if self._meta_controller is not None and hasattr(self._meta_controller, "record_human_feedback"):
            try:
                self._meta_controller.record_human_feedback(
                    task_id=str(task_id or "task"),
                    prompt=str(prompt or ""),
                    agent_response=str(agent_response or ""),
                    correct_response=str(correct_response or "") if correct_response is not None else None,
                    rating=rating,
                    metadata=dict(metadata or {}),
                )
                self._dirty_loops.add("meta_reflection")
            except Exception:
                pass

        # Optional: surface satisfaction/rating as a MetricEvent so the
        # self-evolution stack can consume it via RealTimeMetricsCollector.
        if rating is not None and self._collector is not None:
            norm: float | None = None
            try:
                raw = float(rating)
                if raw <= 1.0:
                    norm = max(0.0, min(1.0, raw))
                elif raw <= 5.0:
                    norm = max(0.0, min(1.0, raw / 5.0))
                else:
                    norm = max(0.0, min(1.0, raw / 10.0))
            except Exception:
                norm = None

            emit = getattr(self._collector, "emit_event", None)
            if callable(emit):
                try:
                    emit(
                        "human_feedback",
                        latency=0.0,
                        energy=0.0,
                        throughput=0.0,
                        status=None,
                        confidence=norm,
                        stage=str(task_id or "task"),
                        metadata={
                            "rating": float(rating),
                            "has_correction": 1.0 if correct_response is not None else 0.0,
                        },
                    )
                except Exception:
                    pass
        return demo

    # ------------------------------------------------------------------
    def register_knowledge_fact(
        self, fact: KnowledgeFact | Dict[str, Any]
    ) -> Dict[str, Any]:
        """Queue a knowledge fact for ingestion and trigger knowledge loop."""

        if is_dataclass(fact):
            payload = asdict(fact)  # type: ignore[arg-type]
        elif isinstance(fact, KnowledgeFact):  # type: ignore[arg-type]
            payload = asdict(fact)
        elif isinstance(fact, dict):
            payload = dict(fact)
        else:
            payload = {
                "value": repr(fact),
            }
        self._pending_facts.append(payload)
        self._dirty_loops.add("knowledge")
        return payload

    # ------------------------------------------------------------------
    def on_regression_detected(self, regression: Dict[str, Any]) -> None:
        """Record a regression signal and mark remediation loops dirty."""

        self._pending_regressions.append(dict(regression))
        self._dirty_loops.update({"skills", "knowledge", "reflection"})

    # ------------------------------------------------------------------
    def notify_architecture_update(self, payload: Dict[str, Any]) -> None:
        """Allow observers to store metadata from architecture updates."""

        self._pending_regressions.append({"architecture_update": payload})
        self._dirty_loops.add("reflection")

    # ------------------------------------------------------------------
    def run_maintenance_cycle(
        self,
        *,
        force: bool = False,
        now: Optional[float] = None,
    ) -> Dict[str, bool]:
        """Execute all loops that are due and return execution flags."""

        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            items = list(self._loops.items())
        results: Dict[str, bool] = {}
        for name, state in items:
            if not state.enabled:
                results[name] = False
                continue
            due = force or name in self._dirty_loops or timestamp - state.last_run >= state.interval
            if not due:
                results[name] = False
                continue
            executed = self._execute_loop(name, state)
            state.last_run = timestamp
            self._dirty_loops.discard(name)
            results[name] = bool(executed)
        return results

    # ------------------------------------------------------------------
    def _execute_loop(self, name: str, state: _LoopState) -> bool:
        callback = state.callback
        if self._executor is not None:
            self._executor(callback, self)
            return True
        result = callback(self)
        return bool(result)

    # ------------------------------------------------------------------
    def start_background_worker(self) -> None:
        """Launch a daemon thread that repeatedly runs maintenance cycles."""

        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event = threading.Event()
        interval = max(1.0, float(self.config.background_interval))

        def _worker() -> None:
            while self._stop_event and not self._stop_event.wait(interval):
                self.run_maintenance_cycle()

        self._worker_thread = threading.Thread(
            target=_worker,
            name="continual-learning-worker",
            daemon=True,
        )
        self._worker_thread.start()

    # ------------------------------------------------------------------
    def shutdown(self, *, wait: bool = False) -> None:
        """Stop the background worker thread if running."""

        if self._stop_event is not None:
            self._stop_event.set()
        if wait and self._worker_thread is not None:
            self._worker_thread.join(timeout=self.config.background_interval)
        self._worker_thread = None
        self._stop_event = None

    # ------------------------------------------------------------------
    # Loop implementations -------------------------------------------------
    # ------------------------------------------------------------------
    def _skill_loop(self, _: "ContinualLearningCoordinator") -> bool:
        if self._policy_trainer is None:
            return False
        dataset = list(self.experience_hub.load_for_training())
        if not dataset:
            dataset = [asdict(record) for record in self.experience_hub.latest()]
        if not dataset:
            return False
        self._policy_trainer(dataset)
        return True

    # ------------------------------------------------------------------
    def _knowledge_loop(self, _: "ContinualLearningCoordinator") -> bool:
        if self._knowledge_importer is None or not self._pending_facts:
            return False
        facts = list(self._pending_facts)
        self._pending_facts.clear()
        try:
            self._knowledge_importer.ingest_facts(facts)
        except Exception:
            # Knowledge ingestion is best-effort; push facts back for retry.
            self._pending_facts[:0] = facts
            return False
        return True

    # ------------------------------------------------------------------
    def _cognition_loop(self, _: "ContinualLearningCoordinator") -> bool:
        if self._cognition is None:
            return False
        feedback: Optional[Iterable[Any]] = None
        if self._collector is not None:
            events = self._collector.events()
            if events:
                feedback = [self._cognition.feedback_from_event(event) for event in events]
        self._cognition.observe(feedback)

        architecture = self._architecture
        cognition = self._cognition
        if (
            architecture is not None
            and hasattr(architecture, "update_architecture")
            and hasattr(cognition, "version")
            and hasattr(architecture, "version")
            and getattr(cognition, "version", 0) > getattr(architecture, "version", -1)
        ):
            metrics = None
            performance = None
            history = getattr(cognition, "history", None)
            if isinstance(history, list) and history:
                last = history[-1]
                performance = getattr(last, "performance", None)
                metrics = getattr(last, "metrics", None)
            try:
                architecture.update_architecture(  # type: ignore[call-arg]
                    getattr(cognition, "architecture", {}),
                    performance=performance,
                    metrics=metrics if isinstance(metrics, dict) else None,
                )
            except Exception:
                pass
        return True

    # ------------------------------------------------------------------
    def _architecture_loop(self, _: "ContinualLearningCoordinator") -> bool:
        if self._architecture is None:
            return False
        regression = self._architecture.run_regression_analysis()
        return bool(regression)

    # ------------------------------------------------------------------
    def _structural_loop(self, _: "ContinualLearningCoordinator") -> bool:
        manager = self._structural_manager
        if manager is None:
            return False
        events = self._collector.events() if self._collector is not None else []
        bottlenecks = self._derive_bottlenecks_from_events(events)
        performance = manager.architecture.evolver.fitness_fn(
            manager.architecture.architecture
        )
        manager.evolve_structure(
            performance=performance,
            bottlenecks=bottlenecks,
            commit=True,
        )
        return True

    # ------------------------------------------------------------------
    def _diagnosis_loop(self, _: "ContinualLearningCoordinator") -> bool:
        diagnoser = self._performance_diagnoser
        if diagnoser is None:
            return False
        events = self._collector.events() if self._collector is not None else []
        report = diagnoser.diagnose(events)
        issues = report.get("issues", [])
        if self._strategy_adjuster is not None:
            current_params: Dict[str, float] = {}
            if self._evolution_engine is not None:
                current_params.update(self._evolution_engine.cognition.architecture)
            adjustments = self._strategy_adjuster.propose(issues, current_params=current_params)
            updates = adjustments.get("updates", {})
            if updates and self._evolution_engine is not None:
                self._evolution_engine.cognition.architecture.update(updates)
        if self._evolution_engine is not None and issues:
            last_version = self._evolution_engine.cognition.version
            if last_version > 0:
                try:
                    self._evolution_engine.rollback(last_version - 1)
                except Exception:
                    pass
        return bool(issues)

    # ------------------------------------------------------------------
    def _imitation_loop(self, _: "ContinualLearningCoordinator") -> bool:
        learner = self._imitation_learner
        if learner is None or self.experience_hub is None:
            return False
        demos = list(self.experience_hub.load_demonstrations())
        if not demos:
            return False
        learner.train(demos)
        return True

    # ------------------------------------------------------------------
    def _representation_loop(self, _: "ContinualLearningCoordinator") -> bool:
        learner = self._representation_learner
        if learner is None:
            return False
        observations: List[Dict[str, Any]] = []
        if self.experience_hub is not None:
            for record in self.experience_hub.latest(limit=10):
                observations.append(record.metadata)
        if not observations:
            return False
        learner.train(observations)
        return True

    # ------------------------------------------------------------------
    def _meta_reflection_loop(self, _: "ContinualLearningCoordinator") -> bool:
        controller = self._meta_controller
        if controller is None:
            return False
        regressions = list(self._pending_regressions)
        if regressions and hasattr(controller, "record_regressions"):
            controller.record_regressions(regressions)
            self._pending_regressions.clear()
        # Optionally ingest recent task outcomes from experience hub.
        if self.experience_hub is not None:
            for record in self.experience_hub.latest(limit=5):
                controller.record_task_outcome(
                    task_id=record.task_id,
                    success=record.success,
                    metadata=record.metadata,
                )
        signals = controller.analyse()
        applied = False
        for signal in signals or []:
            kind = getattr(signal, "kind", None)
            payload = getattr(signal, "payload", None)
            if kind == "schedule_skill_learning":
                self._dirty_loops.add("skills")
                applied = True
            elif kind == "schedule_knowledge_learning":
                self._dirty_loops.add("knowledge")
                applied = True
            elif kind == "trigger_evolution":
                # Nudge evolution/diagnosis loops when meta-cognition detects regressions.
                self._dirty_loops.update({"cognition", "diagnosis"})
                applied = True
            elif kind == "memory_consolidation":
                self._dirty_loops.add("memory_consolidation")
                applied = True
            elif kind == "knowledge_fact" and isinstance(payload, dict):
                fact = payload.get("fact")
                if isinstance(fact, dict):
                    self.register_knowledge_fact(fact)
                    applied = True
            elif kind == "imitation_feedback" and isinstance(payload, dict):
                self._dirty_loops.add("imitation")
                applied = True
        # Heuristic: if signals suggest replan, mark reflection loop dirty.
        if signals and self._reflection_callback is not None:
            self._dirty_loops.add("reflection")
        return bool(signals) or applied

    # ------------------------------------------------------------------
    def _task_generation_loop(self, _: "ContinualLearningCoordinator") -> bool:
        generator = self._task_generator
        if generator is None:
            return False
        state = {"idle": True, "novelty_score": 0.5}
        tasks = generator.generate(state)
        scheduler = self._task_scheduler
        if scheduler is not None:
            scheduler.enqueue([task.__dict__ if hasattr(task, "__dict__") else task for task in tasks])
        if tasks and self._task_callback is not None:
            try:
                self._task_callback([task.__dict__ if hasattr(task, "__dict__") else task for task in tasks])
            except Exception:
                pass
        return bool(tasks)

    # ------------------------------------------------------------------
    def _memory_consolidation_loop(self, _: "ContinualLearningCoordinator") -> bool:
        consolidator = self._memory_consolidator
        if consolidator is None:
            return False
        snapshots: List[Dict[str, Any]] = []
        if self.experience_hub is not None:
            for record in self.experience_hub.latest(limit=5):
                snapshots.append(record.metadata)
        consolidator.consolidate(snapshots)
        return True

    # ------------------------------------------------------------------
    def _derive_bottlenecks_from_events(
        self, events: Iterable["MetricEvent"]
    ) -> List[tuple[str, float]]:
        samples: Dict[str, List[float]] = {}
        for event in events:
            module = getattr(event, "module", None)
            if module is None:
                continue
            samples.setdefault(module, []).append(float(getattr(event, "latency", 0.0)))
        averages = [
            (module, sum(values) / max(len(values), 1))
            for module, values in samples.items()
            if values
        ]
        averages.sort(key=lambda item: item[1], reverse=True)
        return averages

    # ------------------------------------------------------------------
    def _reflection_loop(self, _: "ContinualLearningCoordinator") -> bool:
        if self._reflection_callback is None:
            return False
        payload: Dict[str, Any] = {}
        if self._pending_regressions:
            payload["regressions"] = list(self._pending_regressions)
            self._pending_regressions.clear()

        # Provide lightweight context to reflection callbacks so they can
        # persist structured summaries without needing to capture coordinator
        # state via closures.
        if self.experience_hub is not None:
            try:
                episodes = self.experience_hub.latest(limit=10)
                payload["episodes"] = [asdict(ep) for ep in episodes]
            except Exception:
                pass

        events = self._collector.events() if self._collector is not None else []
        if events:
            payload["metrics_window"] = len(events)
            payload["bottlenecks"] = self._derive_bottlenecks_from_events(events)[:5]

        if self._performance_diagnoser is not None and events:
            try:
                report = self._performance_diagnoser.diagnose(events)
                summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
                issues = report.get("issues", []) if isinstance(report.get("issues"), list) else []
                payload["metrics_summary"] = dict(summary)
                payload["issues"] = [asdict(issue) if is_dataclass(issue) else issue for issue in issues][:12]

                if self._strategy_adjuster is not None and issues:
                    current_params: Dict[str, float] = {}
                    if self._evolution_engine is not None:
                        current_params.update(self._evolution_engine.cognition.architecture)
                    adjustments = self._strategy_adjuster.propose(
                        issues,  # type: ignore[arg-type]
                        current_params=current_params,
                    )
                    actions = adjustments.get("actions", []) if isinstance(adjustments, dict) else []
                    payload["strategy_suggestions"] = {
                        "updates": dict(adjustments.get("updates", {}) if isinstance(adjustments, dict) else {}),
                        "actions": [asdict(action) if is_dataclass(action) else action for action in actions]
                        if isinstance(actions, list)
                        else [],
                    }
            except Exception:
                pass
        self._reflection_callback(payload)
        return True


__all__ = ["ContinualLearningCoordinator", "LearningLoopConfig", "KnowledgeFact"]
