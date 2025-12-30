"""Self-evolving AI architecture orchestrator."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Tuple, TYPE_CHECKING, Callable
from collections import defaultdict, deque

try:  # pragma: no cover - support both repo-root and `modules/` on sys.path
    from modules.monitoring.collector import RealTimeMetricsCollector
except ModuleNotFoundError:  # pragma: no cover
    from monitoring.collector import RealTimeMetricsCollector

if TYPE_CHECKING:
    try:  # pragma: no cover
        from modules.memory.lifecycle import MemoryLifecycleManager
        from modules.brain.state import BrainRuntimeConfig, CuriosityState
    except ModuleNotFoundError:  # pragma: no cover
        from memory.lifecycle import MemoryLifecycleManager
        from brain.state import BrainRuntimeConfig, CuriosityState
else:  # pragma: no cover - typing only
    MemoryLifecycleManager = Any
    BrainRuntimeConfig = Any
    CuriosityState = Any
from .self_evolving_cognition import EvolutionRecord, SelfEvolvingCognition
from .evolving_cognitive_architecture import EvolvingCognitiveArchitecture
try:  # pragma: no cover - optional safety gates
    from .safety import (
        ArchitectureApprovalQueue,
        EvolutionSafetyConfig,
        SafetyDecision,
        PytestSandboxRunner,
        evaluate_candidate,
    )
except Exception:  # pragma: no cover - keep architecture usable without safety module
    ArchitectureApprovalQueue = None  # type: ignore[assignment]
    EvolutionSafetyConfig = None  # type: ignore[assignment]
    SafetyDecision = None  # type: ignore[assignment]
    PytestSandboxRunner = None  # type: ignore[assignment]
    evaluate_candidate = None  # type: ignore[assignment]
from .neuroevolution_backend import (
    CognitiveNetworkGenome,
    NeuroevolutionBackend,
)
from .structural_encoding import encode_structure
from .structural_genome import StructuralGenome

if TYPE_CHECKING:
    from .evolution_recorder import EvolutionKnowledgeRecorder
    try:  # pragma: no cover
        from modules.learning.continual_learning import ContinualLearningCoordinator
    except ModuleNotFoundError:  # pragma: no cover
        from learning.continual_learning import ContinualLearningCoordinator
    from .structural_evolution import StructuralEvolutionManager
else:  # pragma: no cover - typing only
    EvolutionKnowledgeRecorder = Any
    ContinualLearningCoordinator = Any


class SelfEvolvingAIArchitecture:
    """Analyse metrics and evolve an AI architecture accordingly.

    This class cooperates with :class:`SelfEvolvingCognition` and
    :class:`EvolvingCognitiveArchitecture` to share evolution history and provide
    rollback capabilities.  Metrics are collected via
    :class:`RealTimeMetricsCollector`.
    """

    def __init__(
        self,
        initial_architecture: Dict[str, float],
        evolver: EvolvingCognitiveArchitecture,
        collector: Optional[RealTimeMetricsCollector] = None,
        cognition: Optional[SelfEvolvingCognition] = None,
        recorder: Optional[EvolutionKnowledgeRecorder] = None,
        memory_manager: Optional[MemoryLifecycleManager] = None,
        curiosity_state: Optional[CuriosityState] = None,
        brain_config: Optional[BrainRuntimeConfig] = None,
        policy_module: Optional[Any] = None,
        learning_modules: Optional[List[Any]] = None,
        reflection_controller: Optional[Any] = None,
        continual_learning: Optional[ContinualLearningCoordinator] = None,
        structural_manager: Optional["StructuralEvolutionManager"] = None,
        neuro_backend: Optional[NeuroevolutionBackend] = None,
        neuro_generations: int = 1,
        safety_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.collector = collector
        self.evolver = evolver
        self.cognition = cognition
        self.recorder = recorder
        self.memory_manager = memory_manager
        self.curiosity_state = curiosity_state
        self.brain_config = brain_config
        self.policy_module = policy_module
        self._policy_cache: Dict[int, Any] = {}
        self._additional_learning_modules = list(learning_modules or [])
        self.reflection_controller = reflection_controller
        self.continual_learning = continual_learning
        self.structural_manager = structural_manager
        self.neuro_backend = neuro_backend
        self._neuro_generations = max(1, int(neuro_generations))
        self._neuro_struct_version: float = 0.0

        self._defaults = self._derive_default_genome(initial_architecture)
        self._regression_last_version: Optional[int] = None
        self._safety = EvolutionSafetyConfig.from_sources(safety_config) if EvolutionSafetyConfig is not None else None
        self._safety_queue = (
            ArchitectureApprovalQueue(max_pending=self._safety.max_pending_reviews)
            if self._safety is not None and ArchitectureApprovalQueue is not None
            else None
        )
        self._sandbox_runner: Optional[Callable[[], bool]] = None
        if self._safety is not None:
            custom = None
            if isinstance(safety_config, Mapping):
                custom = safety_config.get("sandbox_runner")
                sandbox_cfg = safety_config.get("sandbox")
                if custom is None and isinstance(sandbox_cfg, Mapping):
                    custom = sandbox_cfg.get("runner")
            if callable(custom):
                self._sandbox_runner = custom  # type: ignore[assignment]
            elif PytestSandboxRunner is not None and self._safety.sandbox_pytest_paths:
                self._sandbox_runner = PytestSandboxRunner(
                    paths=self._safety.sandbox_pytest_paths,
                    timeout_s=self._safety.sandbox_timeout_s,
                    extra_args=self._safety.sandbox_pytest_args,
                )
        self._last_safety_decision: Any = None

        base_fitness = getattr(self.evolver, "_behaviour_base_fitness", None)
        if base_fitness is None:
            base_fitness = self.evolver.fitness_fn
            setattr(self.evolver, "_behaviour_base_fitness", base_fitness)
        self._base_fitness_fn = base_fitness
        self.evolver.fitness_fn = self._fitness_with_behaviour
        if hasattr(self.evolver, "ga"):
            self.evolver.ga.fitness_fn = self._fitness_with_behaviour

        if cognition is not None:
            # Share history and version with the cognition module
            self.history = cognition.history
            self.version = cognition.version
            self.architecture = self._merge_with_defaults(cognition.architecture)
            cognition.architecture = self.architecture
            if cognition.history:
                cognition.history[-1].architecture = self.architecture.copy()
        else:
            self.architecture = self._merge_with_defaults(initial_architecture)
            initial_perf = self.evolver.fitness_fn(self.architecture)
            self.version = 0
            metrics = self._collect_behaviour_metrics()
            metrics.setdefault("resource_score", initial_perf)
            self.history: List[EvolutionRecord] = [
                EvolutionRecord(
                    self.version,
                    self.architecture.copy(),
                    initial_perf,
                    metrics,
                )
            ]
        self._apply_architecture_to_modules()

        if self.continual_learning is not None:
            self.continual_learning.attach_architecture(self)
            if cognition is not None:
                self.continual_learning.attach_cognition(cognition)
            if collector is not None:
                self.continual_learning.set_collector(collector)

        regression_annotation = self._prepare_regression_annotations()
        if self.recorder is not None and getattr(self, "history", None):
            annotations = (
                {"regression": regression_annotation}
                if regression_annotation is not None
                else None
            )
            self.recorder.record(
                self.history[0], previous_architecture=None, annotations=annotations
            )

        if self.neuro_backend is None:
            self.neuro_backend = self._build_neuro_backend_from_structure()

    # ------------------------------------------------------------------
    def analyze_performance_bottlenecks(self) -> List[Tuple[str, float]]:
        """Identify modules with highest average latency from collected metrics."""

        if self.collector is None:
            return []
        events = self.collector.events()
        if not events:
            return []

        stats: Dict[str, List[float]] = defaultdict(list)
        for event in events:
            stats[event.module].append(event.latency)
        averages = [(module, sum(vals) / len(vals)) for module, vals in stats.items()]
        averages.sort(key=lambda x: x[1], reverse=True)
        return averages

    # ------------------------------------------------------------------
    def generate_architecture_mutations(
        self, num_candidates: Optional[int] = None
    ) -> List[Tuple[Dict[str, float], float]]:
        """Generate candidate architectures using the genetic algorithm."""

        best, best_score, history = self.evolver.ga.evolve(self.architecture)
        candidates: List[Tuple[Dict[str, float], float]] = list(history)
        if (best, best_score) not in candidates:
            candidates.append((best, best_score))
        if num_candidates is not None:
            candidates = candidates[:num_candidates]
        return candidates

    # ------------------------------------------------------------------
    def evolutionary_selection(
        self, candidates: List[Tuple[Dict[str, float], float]]
    ) -> Dict[str, float]:
        """Select the best candidate based on fitness and update architecture."""

        if not candidates:
            return self.architecture
        best_arch, best_score = max(candidates, key=lambda x: x[1])
        self.update_architecture(best_arch, best_score)
        return best_arch

    # ------------------------------------------------------------------
    def update_architecture(
        self,
        new_arch: Dict[str, float],
        performance: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Apply ``new_arch`` and record it in the evolution history.

        When safety is enabled (see :class:`~modules.evolution.safety.EvolutionSafetyConfig`),
        the update may be rejected or queued for manual review. In that case the
        method returns ``False`` and the current architecture remains unchanged.
        """

        ctx = dict(context or {})
        if ctx.get("skip_safety"):
            self._last_safety_decision = None
            return self._commit_architecture(new_arch, performance=performance, metrics=metrics, context=ctx)

        normalised = self._normalise_architecture({**self.architecture, **new_arch})
        candidate_perf = performance
        if candidate_perf is None:
            candidate_perf = self.evolver.fitness_fn(normalised)

        safety_annotation: Optional[Dict[str, Any]] = None
        decision: Any = None
        safety_cfg = self._safety
        if (
            safety_cfg is not None
            and bool(getattr(safety_cfg, "enabled", False))
            and evaluate_candidate is not None
        ):
            effective_cfg = safety_cfg
            if ctx.get("skip_manual_review") and bool(getattr(safety_cfg, "manual_review_enabled", False)):
                effective_cfg = replace(safety_cfg, manual_review_enabled=False)

            history_perfs: List[float] = []
            for rec in getattr(self, "history", []) or []:
                try:
                    history_perfs.append(float(getattr(rec, "performance", 0.0)))
                except Exception:
                    continue

            decision = evaluate_candidate(
                config=effective_cfg,
                history_performances=history_perfs,
                current_architecture=self.architecture,
                candidate_architecture=normalised,
                candidate_performance=float(candidate_perf),
                approval_queue=self._safety_queue,
                extra_details={"source": ctx.get("source", "update_architecture")},
            )
            self._last_safety_decision = decision
            if getattr(decision, "requires_review", False):
                return False
            if not getattr(decision, "allowed", True):
                return False

            safety_annotation = {
                "decision": getattr(decision, "reason", "accepted"),
                "delta_l1": float(getattr(decision, "delta_l1", 0.0)),
                "baseline_mean": getattr(decision, "baseline_mean", None),
                "baseline_window": getattr(decision, "baseline_window", None),
                "candidate_performance": getattr(decision, "candidate_performance", None),
                "review_id": getattr(decision, "review_id", None),
                "details": getattr(decision, "details", None),
            }
            if ctx.get("manual_review_id") is not None:
                safety_annotation["manual_review_id"] = ctx.get("manual_review_id")

        if (
            safety_cfg is not None
            and bool(getattr(safety_cfg, "sandbox_enabled", False))
            and self._sandbox_runner is not None
            and not ctx.get("skip_sandbox")
        ):
            if not self._run_sandbox_validation(normalised):
                self._last_safety_decision = (
                    SafetyDecision(allowed=False, reason="sandbox_failed") if SafetyDecision is not None else None
                )
                return False
            if safety_annotation is not None:
                safety_annotation["sandbox"] = "passed"

        return self._commit_architecture(
            normalised,
            performance=float(candidate_perf),
            metrics=metrics,
            context=ctx,
            safety_annotation=safety_annotation,
        )

    # ------------------------------------------------------------------
    def rollback(self, version: int) -> Dict[str, float]:
        """Rollback to a previous architecture version."""

        if self.cognition is not None:
            arch = self.cognition.rollback(version)
            self.architecture = arch
            self.version = self.cognition.version
            self._apply_architecture_to_modules()
            return arch
        for record in self.history:
            if record.version == version:
                self.architecture = record.architecture.copy()
                self.version = record.version
                self._apply_architecture_to_modules()
                return self.architecture
        raise ValueError(f"Version {version} not found in history")

    # ------------------------------------------------------------------
    def list_pending_architecture_updates(self, *, limit: int = 50) -> List[Any]:
        if self._safety_queue is None:
            return []
        return list(self._safety_queue.list(limit=limit))

    def approve_pending_architecture_update(self, update_id: str, *, run_sandbox: bool = True) -> bool:
        if self._safety_queue is None:
            return False
        pending = self._safety_queue.get(update_id)
        if pending is None:
            return False
        ctx = {"skip_manual_review": True, "manual_review_id": pending.id, "source": "manual_review_approval"}
        if not run_sandbox:
            ctx["skip_sandbox"] = True
        ok = bool(
            self.update_architecture(
                dict(pending.architecture),
                performance=float(pending.performance),
                metrics=None,
                context=ctx,
            )
        )
        if ok:
            self._safety_queue.pop(update_id)
        return ok

    def deny_pending_architecture_update(self, update_id: str) -> bool:
        if self._safety_queue is None:
            return False
        return self._safety_queue.pop(update_id) is not None

    # ------------------------------------------------------------------
    def _run_sandbox_validation(self, candidate: Dict[str, float]) -> bool:
        runner = self._sandbox_runner
        if runner is None:
            return True
        previous = self.architecture.copy()
        previous_version = self.version
        try:
            self.architecture = dict(candidate)
            self._apply_architecture_to_modules()
            return bool(runner())
        finally:
            self.architecture = previous
            self.version = previous_version
            try:
                self._apply_architecture_to_modules()
            except Exception:
                pass

    def _commit_architecture(
        self,
        new_arch: Dict[str, float],
        *,
        performance: Optional[float],
        metrics: Optional[Dict[str, float]],
        context: Dict[str, Any],
        safety_annotation: Optional[Dict[str, Any]] = None,
    ) -> bool:
        normalised = self._normalise_architecture({**self.architecture, **new_arch})
        self.version += 1
        if performance is None:
            performance = self.evolver.fitness_fn(normalised)
        self.architecture = normalised
        self._apply_architecture_to_modules()
        metrics_payload: Dict[str, Any] = {**(metrics or {})}
        behaviour_metrics = self._collect_behaviour_metrics()
        for key, value in behaviour_metrics.items():
            metrics_payload.setdefault(key, value)
        metrics_payload.setdefault("resource_score", float(performance))
        if safety_annotation is not None:
            metrics_payload.setdefault("safety_delta_l1", float(safety_annotation.get("delta_l1", 0.0) or 0.0))
            baseline = safety_annotation.get("baseline_mean")
            if isinstance(baseline, (int, float)):
                metrics_payload.setdefault("safety_baseline_mean", float(baseline))

        record = EvolutionRecord(
            self.version,
            self.architecture.copy(),
            float(performance),
            dict(metrics_payload),
        )
        self.history.append(record)
        if self.cognition is not None:
            self.cognition.architecture = self.architecture
            self.cognition.version = self.version
        if self.continual_learning is not None:
            try:
                self.continual_learning.notify_architecture_update(record.metrics)
            except Exception:
                pass
        regression_annotation = self._prepare_regression_annotations()
        if self.recorder is not None:
            previous_arch = None
            if len(self.history) >= 2:
                previous_arch = self.history[-2].architecture
            annotations: Dict[str, Any] = {}
            if regression_annotation is not None:
                annotations["regression"] = regression_annotation
            if safety_annotation is not None:
                annotations["safety"] = safety_annotation
            self.recorder.record(
                record,
                previous_architecture=previous_arch,
                annotations=annotations or None,
            )
        return True

    # ------------------------------------------------------------------
    def get_history(self) -> List[EvolutionRecord]:
        """Return the evolution history."""

        return self.history

    # ------------------------------------------------------------------
    def _derive_default_genome(self, seed: Dict[str, float]) -> Dict[str, float]:
        """Build default genome entries informed by attached modules."""

        seed = seed or {}
        defaults: Dict[str, float] = {
            "memory_short_term_limit": float(seed.get("memory_short_term_limit", 25.0)),
            "memory_working_limit": float(seed.get("memory_working_limit", 50.0)),
            "curiosity_drive_floor": float(seed.get("curiosity_drive_floor", 0.4)),
            "curiosity_novelty_preference": float(
                seed.get("curiosity_novelty_preference", 0.5)
            ),
            "curiosity_fatigue_ceiling": float(seed.get("curiosity_fatigue_ceiling", 0.1)),
            "planner_structured_flag": float(seed.get("planner_structured_flag", 1.0)),
            "planner_reinforcement_flag": float(seed.get("planner_reinforcement_flag", 0.0)),
            "policy_learning_rate": float(seed.get("policy_learning_rate", 0.08)),
            "policy_exploration_rate": float(seed.get("policy_exploration_rate", 0.12)),
            "cognitive_policy_variant": float(seed.get("cognitive_policy_variant", 0.0)),
            "planner_min_steps": float(seed.get("planner_min_steps", 4.0)),
            "policy_replay_buffer_size": float(seed.get("policy_replay_buffer_size", 256.0)),
            "policy_replay_batch_size": float(seed.get("policy_replay_batch_size", 16.0)),
            "policy_replay_iterations": float(seed.get("policy_replay_iterations", 1.0)),
            "policy_hidden_dim": float(seed.get("policy_hidden_dim", 128.0)),
            "policy_num_layers": float(seed.get("policy_num_layers", 2.0)),
            "memory_summary_batch_size": float(
                seed.get("memory_summary_batch_size", 5.0)
            ),
            "memory_summary_rate_limit": float(
                seed.get("memory_summary_rate_limit", 900.0)
            ),
            "reflection_interval_hours": float(
                seed.get("reflection_interval_hours", 24.0)
            ),
            "module_self_learning_flag": float(
                seed.get("module_self_learning_flag", 1.0)
            ),
            "module_curiosity_feedback_flag": float(
                seed.get("module_curiosity_feedback_flag", 1.0)
            ),
            "module_metrics_flag": float(seed.get("module_metrics_flag", 1.0)),
        }
        if self.memory_manager is not None:
            defaults["memory_short_term_limit"] = float(
                getattr(self.memory_manager, "_short_term_limit", defaults["memory_short_term_limit"])
            )
            defaults["memory_working_limit"] = float(
                getattr(self.memory_manager, "_working_limit", defaults["memory_working_limit"])
            )
            defaults["memory_summary_batch_size"] = float(
                getattr(
                    self.memory_manager,
                    "_summary_batch_size",
                    defaults["memory_summary_batch_size"],
                )
            )
            defaults["memory_summary_rate_limit"] = float(
                getattr(
                    self.memory_manager,
                    "_summary_rate_limit",
                    defaults["memory_summary_rate_limit"],
                )
            )
        if self.curiosity_state is not None:
            defaults["curiosity_drive_floor"] = float(self.curiosity_state.drive)
            defaults["curiosity_novelty_preference"] = float(
                self.curiosity_state.novelty_preference
            )
            defaults["curiosity_fatigue_ceiling"] = float(self.curiosity_state.fatigue)
        if self.brain_config is not None:
            defaults["planner_structured_flag"] = (
                1.0
                if getattr(self.brain_config, "prefer_structured_planner", True)
                else 0.0
            )
            defaults["planner_reinforcement_flag"] = (
                1.0
                if getattr(self.brain_config, "prefer_reinforcement_planner", False)
                else 0.0
            )
            defaults["module_self_learning_flag"] = (
                1.0 if getattr(self.brain_config, "enable_self_learning", True) else 0.0
            )
            defaults["module_curiosity_feedback_flag"] = (
                1.0
                if getattr(self.brain_config, "enable_curiosity_feedback", True)
                else 0.0
            )
            defaults["module_metrics_flag"] = (
                1.0 if getattr(self.brain_config, "metrics_enabled", True) else 0.0
            )
            if "cognitive_policy_variant" not in seed:
                defaults["cognitive_policy_variant"] = (
                    1.0 if getattr(self.brain_config, "prefer_reinforcement_planner", False) else 0.0
                )

        current_policy: Any | None = None
        if self.policy_module is not None:
            current_policy = getattr(self.policy_module, "policy", None)
            if current_policy is None:
                current_policy = self.policy_module

        if current_policy is not None:
            if "cognitive_policy_variant" not in seed:
                policy_name = type(current_policy).__name__
                if "Bandit" in policy_name:
                    defaults["cognitive_policy_variant"] = 2.0
                elif "Reinforcement" in policy_name:
                    defaults["cognitive_policy_variant"] = 1.0

            if "planner_min_steps" not in seed:
                planner = getattr(current_policy, "planner", None)
                min_steps = getattr(planner, "min_steps", None)
                if isinstance(min_steps, int):
                    defaults["planner_min_steps"] = float(min_steps)

            if "policy_replay_buffer_size" not in seed:
                buffer = getattr(current_policy, "_experience_buffer", None)
                maxlen = getattr(buffer, "maxlen", None) if buffer is not None else None
                if isinstance(maxlen, int) and maxlen > 0:
                    defaults["policy_replay_buffer_size"] = float(maxlen)
            if "policy_replay_batch_size" not in seed:
                batch_size = getattr(current_policy, "_replay_batch_size", None)
                if isinstance(batch_size, int) and batch_size > 0:
                    defaults["policy_replay_batch_size"] = float(batch_size)
            if "policy_replay_iterations" not in seed:
                iterations = getattr(current_policy, "_replay_iterations", None)
                if isinstance(iterations, int) and iterations > 0:
                    defaults["policy_replay_iterations"] = float(iterations)
        policy_targets = self._iter_learning_targets()
        for target in policy_targets:
            learning_rate = getattr(target, "learning_rate", None)
            if learning_rate is not None:
                defaults["policy_learning_rate"] = float(learning_rate)
                break
        for target in policy_targets:
            exploration = getattr(target, "exploration", None)
            if exploration is not None:
                defaults["policy_exploration_rate"] = float(exploration)
                break
        for target in policy_targets:
            cfg = getattr(target, "config", None)
            hidden_dim = getattr(cfg, "hidden_dim", None) if cfg is not None else None
            if hidden_dim is not None:
                try:
                    defaults["policy_hidden_dim"] = float(hidden_dim)
                except (TypeError, ValueError):
                    pass
                break
        for target in policy_targets:
            cfg = getattr(target, "config", None)
            num_layers = getattr(cfg, "num_layers", None) if cfg is not None else None
            if num_layers is not None:
                try:
                    defaults["policy_num_layers"] = float(num_layers)
                except (TypeError, ValueError):
                    pass
                break
        if self.reflection_controller is not None:
            interval = getattr(
                self.reflection_controller,
                "reflection_interval_hours",
                getattr(self.reflection_controller, "interval_hours", None),
            )
            if interval is not None:
                try:
                    defaults["reflection_interval_hours"] = float(interval)
                except (TypeError, ValueError):
                    pass
        return defaults

    # ------------------------------------------------------------------
    def _merge_with_defaults(self, arch: Dict[str, float]) -> Dict[str, float]:
        merged = dict(self._defaults)
        merged.update(arch or {})
        return self._normalise_architecture(merged)

    # ------------------------------------------------------------------
    def _normalise_architecture(self, arch: Dict[str, float]) -> Dict[str, float]:
        normalised = dict(arch)
        short_term = float(
            normalised.get(
                "memory_short_term_limit",
                self._defaults.get("memory_short_term_limit", 25.0),
            )
        )
        short_term = max(1.0, short_term)
        normalised["memory_short_term_limit"] = short_term
        working = float(
            normalised.get(
                "memory_working_limit", self._defaults.get("memory_working_limit", 50.0)
            )
        )
        working = max(short_term, working)
        normalised["memory_working_limit"] = working
        for key in (
            "curiosity_drive_floor",
            "curiosity_novelty_preference",
            "curiosity_fatigue_ceiling",
        ):
            value = float(normalised.get(key, self._defaults.get(key, 0.0)))
            normalised[key] = max(0.0, min(1.0, value))
        structured_raw = float(
            normalised.get(
                "planner_structured_flag", self._defaults.get("planner_structured_flag", 1.0)
            )
        )
        reinforcement_raw = float(
            normalised.get(
                "planner_reinforcement_flag",
                self._defaults.get("planner_reinforcement_flag", 0.0),
            )
        )
        normalised["planner_structured_flag"] = 1.0 if structured_raw >= 0.5 else 0.0
        normalised["planner_reinforcement_flag"] = 1.0 if reinforcement_raw >= 0.5 else 0.0
        learning_rate = float(
            normalised.get(
                "policy_learning_rate", self._defaults.get("policy_learning_rate", 0.08)
            )
        )
        learning_rate = max(1e-5, min(1.0, learning_rate))
        normalised["policy_learning_rate"] = learning_rate
        exploration = float(
            normalised.get(
                "policy_exploration_rate",
                self._defaults.get("policy_exploration_rate", 0.12),
            )
        )
        exploration = max(0.0, min(1.0, exploration))
        normalised["policy_exploration_rate"] = exploration
        summary_batch = int(
            round(
                float(
                    normalised.get(
                        "memory_summary_batch_size",
                        self._defaults.get("memory_summary_batch_size", 5.0),
                    )
                )
            )
        )
        normalised["memory_summary_batch_size"] = float(max(1, summary_batch))
        summary_rate = float(
            normalised.get(
                "memory_summary_rate_limit",
                self._defaults.get("memory_summary_rate_limit", 900.0),
            )
        )
        normalised["memory_summary_rate_limit"] = max(60.0, summary_rate)
        reflection_interval = float(
            normalised.get(
                "reflection_interval_hours",
                self._defaults.get("reflection_interval_hours", 24.0),
            )
        )
        normalised["reflection_interval_hours"] = max(0.25, reflection_interval)
        for flag_key in (
            "module_self_learning_flag",
            "module_curiosity_feedback_flag",
            "module_metrics_flag",
        ):
            raw_value = float(normalised.get(flag_key, self._defaults.get(flag_key, 0.0)))
            normalised[flag_key] = 1.0 if raw_value >= 0.5 else 0.0

        variant = int(
            round(
                float(
                    normalised.get(
                        "cognitive_policy_variant",
                        self._defaults.get("cognitive_policy_variant", 0.0),
                    )
                )
            )
        )
        variant = max(0, min(2, variant))
        normalised["cognitive_policy_variant"] = float(variant)

        planner_steps = int(
            round(
                float(
                    normalised.get(
                        "planner_min_steps",
                        self._defaults.get("planner_min_steps", 4.0),
                    )
                )
            )
        )
        normalised["planner_min_steps"] = float(max(1, min(16, planner_steps)))

        buffer_size = int(
            round(
                float(
                    normalised.get(
                        "policy_replay_buffer_size",
                        self._defaults.get("policy_replay_buffer_size", 256.0),
                    )
                )
            )
        )
        normalised["policy_replay_buffer_size"] = float(max(32, min(4096, buffer_size)))

        batch_size = int(
            round(
                float(
                    normalised.get(
                        "policy_replay_batch_size",
                        self._defaults.get("policy_replay_batch_size", 16.0),
                    )
                )
            )
        )
        normalised["policy_replay_batch_size"] = float(max(1, min(256, batch_size)))

        iterations = int(
            round(
                float(
                    normalised.get(
                        "policy_replay_iterations",
                        self._defaults.get("policy_replay_iterations", 1.0),
                    )
                )
            )
        )
        normalised["policy_replay_iterations"] = float(max(1, min(12, iterations)))

        hidden_dim = int(
            round(
                float(
                    normalised.get(
                        "policy_hidden_dim",
                        self._defaults.get("policy_hidden_dim", 128.0),
                    )
                )
            )
        )
        normalised["policy_hidden_dim"] = float(max(8, min(2048, hidden_dim)))

        num_layers = int(
            round(
                float(
                    normalised.get(
                        "policy_num_layers",
                        self._defaults.get("policy_num_layers", 2.0),
                    )
                )
            )
        )
        normalised["policy_num_layers"] = float(max(1, min(8, num_layers)))
        return normalised

    # ------------------------------------------------------------------
    def _iter_learning_targets(self) -> List[Any]:
        """Return attached modules that expose learning hyper-parameters."""

        seen: set[int] = set()
        targets: List[Any] = []

        def _add(candidate: Any) -> None:
            if candidate is None:
                return
            identifier = id(candidate)
            if identifier in seen:
                return
            seen.add(identifier)
            targets.append(candidate)

        _add(getattr(self, "policy_module", None))
        policy = getattr(getattr(self, "policy_module", None), "policy", None)
        _add(policy)
        for module in getattr(self, "_additional_learning_modules", []):
            _add(module)
            nested = getattr(module, "policy", None)
            _add(nested)
        return targets

    # ------------------------------------------------------------------
    def _build_neuro_backend_from_structure(self) -> Optional[NeuroevolutionBackend]:
        if self.structural_manager is None:
            return None
        base_genome = getattr(self.structural_manager, "_genome", None)
        if isinstance(base_genome, StructuralGenome):
            base_genome = CognitiveNetworkGenome(structural=base_genome.clone())
        if base_genome is None:
            topology = getattr(self.structural_manager, "topology", {})
            gates = getattr(self.structural_manager, "module_gates", {})
            base_genome = CognitiveNetworkGenome.from_topology(topology, gates)
        return NeuroevolutionBackend(self._neuro_fitness, base_genome=base_genome)

    # ------------------------------------------------------------------
    def attach_structural_manager(self, manager: "StructuralEvolutionManager") -> None:
        self.structural_manager = manager
        if self.neuro_backend is None:
            self.neuro_backend = self._build_neuro_backend_from_structure()

    # ------------------------------------------------------------------
    def _neuro_fitness(self, genome: CognitiveNetworkGenome) -> float:
        topology, gates = genome.to_topology()
        encoded = encode_structure(topology, gates)
        merged = {**getattr(self, "architecture", {}), **encoded}
        return float(self.evolver.fitness_fn(merged))

    # ------------------------------------------------------------------
    def run_neuroevolution_cycle(
        self,
        *,
        performance: Optional[float] = None,
        generations: Optional[int] = None,
        reason: str = "neuroevolution cycle",
    ) -> Optional[CognitiveNetworkGenome]:
        """Evolve neural topology genomes and apply the winner."""

        if self.neuro_backend is None:
            return None
        best, fitness = self.neuro_backend.evolve(generations or self._neuro_generations)
        perf_score = performance if performance is not None else fitness
        metrics = {"neuro_fitness": float(fitness)}
        if self.structural_manager is not None:
            self.structural_manager.apply_neat_genome(
                best.structural,
                performance=float(perf_score),
                reason=reason,
                extra_metrics=metrics,
            )
        else:
            topology, gates = best.to_topology()
            encoded = encode_structure(topology, gates)
            self._neuro_struct_version += 1.0
            encoded["structural_version"] = float(self._neuro_struct_version)
            self.update_architecture(
                encoded,
                performance=float(perf_score),
                metrics=metrics,
            )
        return best

    # ------------------------------------------------------------------
    def _apply_architecture_to_modules(self) -> None:
        if not hasattr(self, "architecture"):
            return
        arch = self.architecture

        policy_container = getattr(self, "policy_module", None)
        if (
            policy_container is not None
            and hasattr(policy_container, "set_policy")
            and hasattr(policy_container, "policy")
        ):
            variant = int(round(float(arch.get("cognitive_policy_variant", 0.0))))
            planner_min_steps = int(round(float(arch.get("planner_min_steps", 4.0))))
            replay_buffer_size = int(round(float(arch.get("policy_replay_buffer_size", 256.0))))
            replay_batch_size = int(round(float(arch.get("policy_replay_batch_size", 16.0))))
            replay_iterations = int(round(float(arch.get("policy_replay_iterations", 1.0))))
            learning_rate = float(arch.get("policy_learning_rate", self._defaults.get("policy_learning_rate", 0.08)))
            exploration = float(
                arch.get("policy_exploration_rate", self._defaults.get("policy_exploration_rate", 0.12))
            )

            try:  # pragma: no cover - optional brain policy stack
                from modules.brain.whole_brain_policy import (
                    BanditCognitivePolicy,
                    ProductionCognitivePolicy,
                    ReinforcementCognitivePolicy,
                    StructuredPlanner,
                )
            except Exception:  # pragma: no cover - optional dependency missing
                BanditCognitivePolicy = None  # type: ignore[assignment]
                ProductionCognitivePolicy = None  # type: ignore[assignment]
                ReinforcementCognitivePolicy = None  # type: ignore[assignment]
                StructuredPlanner = None  # type: ignore[assignment]

            current_policy = getattr(policy_container, "policy", None)
            if (
                current_policy is not None
                and StructuredPlanner is not None
                and ProductionCognitivePolicy is not None
                and ReinforcementCognitivePolicy is not None
            ):
                planner = getattr(current_policy, "planner", None)
                if planner is None or not hasattr(planner, "min_steps"):
                    planner = StructuredPlanner(min_steps=planner_min_steps)
                else:
                    try:
                        planner.min_steps = max(1, int(planner_min_steps))
                    except Exception:
                        pass

                desired_policy: Any | None = None
                desired_type: Any = ProductionCognitivePolicy
                cache_key = int(variant)
                if variant == 2 and BanditCognitivePolicy is not None:
                    desired_type = BanditCognitivePolicy
                elif variant == 1:
                    desired_type = ReinforcementCognitivePolicy

                cached = self._policy_cache.get(cache_key)
                if cached is not None and isinstance(cached, desired_type):
                    desired_policy = cached
                else:
                    try:
                        if desired_type is ReinforcementCognitivePolicy:
                            desired_policy = ReinforcementCognitivePolicy(
                                learning_rate=learning_rate,
                                exploration=exploration,
                                planner=planner,
                                replay_buffer_size=replay_buffer_size,
                                replay_batch_size=replay_batch_size,
                                replay_iterations=replay_iterations,
                            )
                        elif desired_type is BanditCognitivePolicy and BanditCognitivePolicy is not None:
                            desired_policy = BanditCognitivePolicy(
                                exploration=exploration,
                                planner=planner,
                                fallback=ProductionCognitivePolicy(planner=planner),
                            )
                        else:
                            desired_policy = ProductionCognitivePolicy(planner=planner)
                    except Exception:
                        desired_policy = None

                    if desired_policy is not None:
                        self._policy_cache[cache_key] = desired_policy

                if desired_policy is not None and not isinstance(current_policy, desired_type):
                    try:
                        policy_container.set_policy(desired_policy)
                        current_policy = desired_policy
                    except Exception:
                        pass
        if self.memory_manager is not None:
            short_limit = int(round(arch.get("memory_short_term_limit", 0.0)))
            working_limit = int(round(arch.get("memory_working_limit", short_limit)))
            self.memory_manager._short_term_limit = max(1, short_limit)
            self.memory_manager._working_limit = max(1, working_limit)
            summary_batch = int(
                round(arch.get("memory_summary_batch_size", getattr(self.memory_manager, "_summary_batch_size", 5)))
            )
            summary_rate = float(
                arch.get(
                    "memory_summary_rate_limit",
                    getattr(self.memory_manager, "_summary_rate_limit", 900.0),
                )
            )
            if hasattr(self.memory_manager, "_summary_batch_size"):
                self.memory_manager._summary_batch_size = max(1, summary_batch)
            if hasattr(self.memory_manager, "_summary_rate_limit"):
                self.memory_manager._summary_rate_limit = max(60.0, summary_rate)
        if self.curiosity_state is not None:
            self.curiosity_state.drive = arch.get(
                "curiosity_drive_floor", self.curiosity_state.drive
            )
            self.curiosity_state.novelty_preference = arch.get(
                "curiosity_novelty_preference", self.curiosity_state.novelty_preference
            )
            self.curiosity_state.fatigue = arch.get(
                "curiosity_fatigue_ceiling", self.curiosity_state.fatigue
            )
        if self.brain_config is not None:
            structured = bool(arch.get("planner_structured_flag", 1.0) >= 0.5)
            reinforcement = bool(arch.get("planner_reinforcement_flag", 0.0) >= 0.5)
            self.brain_config.prefer_structured_planner = structured
            self.brain_config.prefer_reinforcement_planner = reinforcement
            self.brain_config.enable_plan_logging = structured
            if "module_self_learning_flag" in arch:
                self.brain_config.enable_self_learning = bool(
                    arch.get("module_self_learning_flag", 1.0) >= 0.5
                )
            if "module_curiosity_feedback_flag" in arch:
                self.brain_config.enable_curiosity_feedback = bool(
                    arch.get("module_curiosity_feedback_flag", 1.0) >= 0.5
                )
            if "module_metrics_flag" in arch:
                self.brain_config.metrics_enabled = bool(
                    arch.get("module_metrics_flag", 1.0) >= 0.5
                )
        learning_targets = self._iter_learning_targets()
        if learning_targets:
            learning_rate = arch.get(
                "policy_learning_rate", self._defaults.get("policy_learning_rate", 0.08)
            )
            exploration = arch.get(
                "policy_exploration_rate",
                self._defaults.get("policy_exploration_rate", 0.12),
            )
            for target in learning_targets:
                if hasattr(target, "learning_rate"):
                    try:
                        target.learning_rate = float(learning_rate)
                    except (TypeError, ValueError):  # pragma: no cover - defensive
                        continue
                if hasattr(target, "exploration"):
                    try:
                        target.exploration = float(exploration)
                    except (TypeError, ValueError):  # pragma: no cover - defensive
                        continue

            planner_min_steps = int(round(float(arch.get("planner_min_steps", 4.0))))
            for target in learning_targets:
                planner = getattr(target, "planner", None)
                if planner is not None and hasattr(planner, "min_steps"):
                    try:
                        planner.min_steps = max(1, int(planner_min_steps))
                    except Exception:
                        continue

            replay_buffer_size = int(round(float(arch.get("policy_replay_buffer_size", 256.0))))
            replay_batch_size = int(round(float(arch.get("policy_replay_batch_size", 16.0))))
            replay_iterations = int(round(float(arch.get("policy_replay_iterations", 1.0))))
            for target in learning_targets:
                if hasattr(target, "_experience_buffer"):
                    buffer = getattr(target, "_experience_buffer", None)
                    maxlen = getattr(buffer, "maxlen", None) if buffer is not None else None
                    if isinstance(maxlen, int) and maxlen != replay_buffer_size:
                        items = list(buffer) if buffer is not None else []
                        trimmed = items[-replay_buffer_size:] if replay_buffer_size > 0 else []
                        try:
                            target._experience_buffer = deque(trimmed, maxlen=max(1, replay_buffer_size))
                        except Exception:
                            pass
                if hasattr(target, "_replay_batch_size"):
                    try:
                        target._replay_batch_size = max(1, int(replay_batch_size))
                    except Exception:
                        pass
                if hasattr(target, "_replay_iterations"):
                    try:
                        target._replay_iterations = max(1, int(replay_iterations))
                    except Exception:
                        pass

            hidden_dim = int(round(float(arch.get("policy_hidden_dim", 128.0))))
            num_layers = int(round(float(arch.get("policy_num_layers", 2.0))))
            for target in learning_targets:
                if hasattr(target, "update_architecture"):
                    try:
                        target.update_architecture(hidden_dim=hidden_dim, num_layers=num_layers)
                    except TypeError:
                        continue
                    except Exception:
                        continue
        if self.reflection_controller is not None and "reflection_interval_hours" in arch:
            interval = float(arch.get("reflection_interval_hours", 24.0))
            controller = self.reflection_controller
            if hasattr(controller, "set_reflection_interval"):
                try:
                    controller.set_reflection_interval(interval)
                except TypeError:  # pragma: no cover - fallback to attribute set
                    setattr(controller, "reflection_interval_hours", interval)
            elif hasattr(controller, "update_interval"):
                controller.update_interval(interval)
            else:
                setattr(controller, "reflection_interval_hours", interval)

    # ------------------------------------------------------------------
    def _prepare_regression_annotations(self) -> Optional[Dict[str, Any]]:
        """Detect regressions and coordinate remediation planning."""

        regression = self._detect_regressions()
        if regression is None:
            return None
        interventions = self._trigger_learning_program(regression)
        regression["interventions"] = interventions
        return regression

    # ------------------------------------------------------------------
    def _detect_regressions(self) -> Optional[Dict[str, Any]]:
        """Inspect recent history for resource or success regressions."""

        history = getattr(self, "history", None)
        if not history or len(history) < 3:
            return None
        window = history[-3:]
        latest_version = window[-1].version
        if self._regression_last_version == latest_version:
            return None

        resource_scores: List[float] = []
        for record in window:
            score = record.metrics.get("resource_score")
            if score is None:
                score = record.performance
            try:
                resource_scores.append(float(score))
            except (TypeError, ValueError):
                resource_scores.append(0.0)
        previous_scores = resource_scores[:-1]
        if not previous_scores:
            return None
        previous_average = sum(previous_scores) / len(previous_scores)
        latest_score = resource_scores[-1]
        drop = previous_average - latest_score
        baseline = previous_average if previous_average != 0 else 1.0
        drop_percentage = drop / baseline if baseline else 0.0
        significant_drop = previous_average > 0 and drop_percentage >= 0.15
        absolute_drop = drop >= 0.1

        success_rates: List[float] = []
        for record in window:
            success = record.metrics.get("success_rate")
            if success is None:
                break
            try:
                success_rates.append(float(success))
            except (TypeError, ValueError):
                break
        success_rate_average: Optional[float] = None
        sustained_low = False
        if len(success_rates) == len(window):
            success_rate_average = sum(success_rates) / len(success_rates)
            low_count = sum(1 for rate in success_rates if rate < 0.5)
            sustained_low = success_rate_average < 0.45 and low_count >= 2

        if not (significant_drop or absolute_drop or sustained_low):
            return None

        reasons: List[str] = []
        if significant_drop or absolute_drop:
            reasons.append("resource_decline")
        if sustained_low:
            reasons.append("low_success_rate")

        regression_details: Dict[str, Any] = {
            "versions": [record.version for record in window],
            "latest_version": latest_version,
            "resource_scores": resource_scores,
            "previous_average": previous_average,
            "latest_score": latest_score,
            "drop": drop,
            "drop_percentage": drop_percentage,
            "reasons": reasons,
        }
        if success_rates:
            regression_details["success_rates"] = success_rates
        if success_rate_average is not None:
            regression_details["success_rate_average"] = success_rate_average
        return regression_details

    # ------------------------------------------------------------------
    def _trigger_learning_program(self, regression: Dict[str, Any]) -> List[str]:
        """Notify attached modules to address detected regressions."""

        interventions: List[str] = []
        reason = "architecture_regression"

        def _invoke(target: Any, label: str, method_names: List[str]) -> bool:
            for method_name in method_names:
                callback = getattr(target, method_name, None)
                if not callable(callback):
                    continue
                try:
                    callback(reason=reason, metadata=regression)
                except TypeError:
                    try:
                        callback(regression)
                    except TypeError:
                        callback()
                interventions.append(f"{label}.{method_name}")
                return True
            return False

        if self.memory_manager is not None:
            handled = _invoke(
                self.memory_manager,
                "memory_manager",
                [
                    "schedule_review",
                    "schedule_memory_review",
                    "request_memory_consolidation",
                ],
            )
            if not handled:
                setattr(self.memory_manager, "needs_review", True)
                interventions.append("memory_manager.needs_review_flag")

        policy_targets = [self.policy_module]
        if self.policy_module is not None:
            nested = getattr(self.policy_module, "policy", None)
            if nested is not None:
                policy_targets.append(nested)
        for target in policy_targets:
            if target is None:
                continue
            handled = _invoke(
                target,
                "policy_module",
                [
                    "schedule_additional_training",
                    "trigger_retraining",
                    "increase_training_intensity",
                ],
            )
            if handled:
                break

        for module in getattr(self, "_additional_learning_modules", []):
            _invoke(
                module,
                module.__class__.__name__,
                [
                    "schedule_practice",
                    "schedule_review",
                    "request_practice",
                ],
            )

        if self.reflection_controller is not None:
            handled = _invoke(
                self.reflection_controller,
                "reflection_controller",
                [
                    "schedule_reflection",
                    "trigger_reflection",
                    "request_reflection",
                ],
            )
            if not handled:
                setattr(self.reflection_controller, "needs_reflection", True)
                interventions.append("reflection_controller.needs_reflection_flag")

        latest_version = regression.get("latest_version")
        if isinstance(latest_version, int):
            self._regression_last_version = latest_version
        if self.continual_learning is not None:
            self.continual_learning.on_regression_detected(regression)
        return interventions

    # ------------------------------------------------------------------
    def run_regression_analysis(self) -> Optional[Dict[str, Any]]:
        """Expose regression detection to external coordinators."""

        return self._prepare_regression_annotations()

    # ------------------------------------------------------------------
    def _collect_behaviour_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if self.collector is None:
            return metrics
        events = self.collector.events()
        if events:
            avg_latency = sum(event.latency for event in events) / len(events)
            avg_throughput = sum(event.throughput for event in events) / len(events)
            avg_energy = sum(event.energy for event in events) / len(events)
            metrics.update(
                {
                    "avg_latency": float(avg_latency),
                    "avg_throughput": float(avg_throughput),
                    "avg_energy": float(avg_energy),
                }
            )
            outcomes = [
                event.status
                if event.status is not None
                else (
                    "success"
                    if event.prediction is not None
                    and event.actual is not None
                    and event.prediction == event.actual
                    else (
                        "failure"
                        if event.prediction is not None
                        and event.actual is not None
                        else None
                    )
                )
                for event in events
            ]
            filtered = [status for status in outcomes if status is not None]
            if filtered:
                success_count = sum(1 for status in filtered if status == "success")
                metrics["success_rate"] = success_count / len(filtered)
        monitor = getattr(self.collector, "_monitor", None)
        storage = getattr(monitor, "storage", None) if monitor is not None else None
        if "success_rate" not in metrics and storage is not None:
            try:
                success_rate = float(storage.success_rate())
            except Exception:
                success_rate = 0.0
            metrics["success_rate"] = success_rate
        return metrics

    # ------------------------------------------------------------------
    def _fitness_with_behaviour(self, architecture: Dict[str, float]) -> float:
        base_score = self._base_fitness_fn(architecture)
        metrics = self._collect_behaviour_metrics()
        if not metrics:
            return base_score
        success_rate = metrics.get("success_rate", 0.0)
        avg_throughput = metrics.get("avg_throughput", 0.0)
        avg_latency = metrics.get("avg_latency", 0.0)
        avg_energy = metrics.get("avg_energy", 0.0)
        score = base_score
        score += success_rate * 2.0
        score += avg_throughput * 0.1
        score -= avg_latency * 0.5
        score -= avg_energy * 0.1
        return score
