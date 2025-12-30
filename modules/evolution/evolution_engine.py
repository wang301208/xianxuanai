from __future__ import annotations

"""Unified evolution engine coordinating cognitive evolution components.

This module defines :class:`EvolutionEngine` which ties together
:class:`SelfEvolvingCognition` and :class:`EvolvingCognitiveArchitecture`.
It exposes :meth:`run_evolution_cycle` to drive the evolution process using
performance metrics and keeps a history of all evolved architectures to enable
rollback.  The engine has been extended with a lightweight specialist module
registry that allows expert solvers to be plugged in for specific task
capabilities.  During an evolution cycle the engine can query the registry and
select a super-human specialist when one matches the current task context.
"""

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, TYPE_CHECKING

try:  # pragma: no cover - support both repo-root and `modules/` on sys.path
    from modules.monitoring.collector import MetricEvent
except ModuleNotFoundError:  # pragma: no cover
    from monitoring.collector import MetricEvent

from .cognitive_benchmark import (
    CognitiveBenchmarkResult,
    aggregate_benchmark_score,
    summarise_benchmarks,
)
from .evolving_cognitive_architecture import EvolvingCognitiveArchitecture, GeneticAlgorithm
from .structural_evolution import StructuralEvolutionManager, StructuralProposal
from .self_evolving_cognition import EvolutionRecord, SelfEvolvingCognition
from .nas import NASMutationSpace
try:  # pragma: no cover - optional meta-NAS controller
    from .meta_nas import MetaNASController
except Exception:  # pragma: no cover
    MetaNASController = None  # type: ignore[assignment]
from .evolution_recorder import EvolutionKnowledgeRecorder
from .multiobjective import adjust_performance
try:  # pragma: no cover - optional safety gates
    from .safety import (
        ArchitectureApprovalQueue,
        EvolutionSafetyConfig,
        SafetyDecision,
        PytestSandboxRunner,
        evaluate_candidate,
    )
except Exception:  # pragma: no cover
    ArchitectureApprovalQueue = None  # type: ignore[assignment]
    EvolutionSafetyConfig = None  # type: ignore[assignment]
    SafetyDecision = None  # type: ignore[assignment]
    PytestSandboxRunner = None  # type: ignore[assignment]
    evaluate_candidate = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing only
    try:  # pragma: no cover
        from modules.monitoring.performance_diagnoser import PerformanceDiagnoser, DiagnosticIssue
    except ModuleNotFoundError:  # pragma: no cover
        from monitoring.performance_diagnoser import PerformanceDiagnoser, DiagnosticIssue
    from .strategy_adjuster import StrategyAdjuster


@dataclass(slots=True)
class TaskContext:
    """Description of a task presented to the evolution engine."""

    name: str
    required_capabilities: Sequence[str] = field(default_factory=tuple)
    metadata: Optional[Mapping[str, Any]] = None

    def requirement_set(self) -> set[str]:
        """Return lower-case capability identifiers for matching specialists."""

        return {cap.lower() for cap in self.required_capabilities}


@dataclass(slots=True)
class SpecialistModule:
    """Metadata wrapper describing a specialist solver module."""

    name: str
    capabilities: set[str]
    solver: Callable[[Dict[str, float], TaskContext], Dict[str, float]]
    priority: float = 0.0
    usage_count: int = 0
    total_score: float = 0.0

    def __post_init__(self) -> None:
        self.capabilities = {cap.lower() for cap in self.capabilities}

    def matches(self, requirements: Iterable[str]) -> bool:
        """Return ``True`` when the specialist covers the given requirements."""

        if not requirements:
            return False
        return set(requirements).issubset(self.capabilities)

    def record_performance(self, score: float) -> None:
        """Record performance feedback for future selection heuristics."""

        self.usage_count += 1
        self.total_score += score

    @property
    def average_score(self) -> float:
        """Return the mean recorded score for the specialist."""

        if self.usage_count == 0:
            return 0.0
        return self.total_score / self.usage_count


class SpecialistModuleRegistry:
    """Registry containing expert modules that can assist evolution."""

    def __init__(self, modules: Optional[Iterable[SpecialistModule]] = None) -> None:
        self._modules: Dict[str, SpecialistModule] = {}
        if modules:
            for module in modules:
                self.register(module)

    def register(self, module: SpecialistModule) -> None:
        """Register or replace a specialist module in the registry."""

        self._modules[module.name] = module

    def get(self, name: str) -> Optional[SpecialistModule]:
        """Return a specialist by name when present."""

        return self._modules.get(name)

    def matching_modules(self, task: TaskContext) -> List[SpecialistModule]:
        """Return all specialists capable of handling ``task``."""

        requirements = task.requirement_set()
        return [
            module
            for module in self._modules.values()
            if module.matches(requirements)
        ]

    def select_best(self, task: TaskContext) -> Optional[SpecialistModule]:
        """Select the most promising specialist for ``task``."""

        candidates = self.matching_modules(task)
        if not candidates:
            return None

        def score(candidate: SpecialistModule) -> tuple[float, float]:
            if candidate.usage_count:
                return (candidate.average_score, candidate.priority)
            return (candidate.priority, candidate.priority)

        return max(candidates, key=score)

    def update_performance(self, name: str, score: float) -> None:
        """Record feedback for a specialist when available."""

        module = self.get(name)
        if module is not None:
            module.record_performance(score)


class EvolutionEngine:
    """Coordinate architecture evolution based on performance metrics."""

    def __init__(
        self,
        initial_architecture: Dict[str, float],
        fitness_fn,
        ga: GeneticAlgorithm | None = None,
        score_weights: Optional[Dict[str, float]] = None,
        nas_space: Optional[NASMutationSpace] = None,
        nas_controller: Optional[Any] = None,
        recorder: Optional[EvolutionKnowledgeRecorder] = None,
        specialist_modules: Optional[Iterable[SpecialistModule]] = None,
        structural_manager: Optional[StructuralEvolutionManager] = None,
        structural_interval: Optional[int] = 1,
        performance_diagnoser: Optional["PerformanceDiagnoser"] = None,
        strategy_adjuster: Optional["StrategyAdjuster"] = None,
        enforce_elite: bool = False,
        regression_tolerance: float = 0.0,
        safety_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        post_mutation = nas_space.postprocess if nas_space is not None else None
        self.evolver = EvolvingCognitiveArchitecture(
            fitness_fn,
            ga,
            post_mutation=post_mutation,
            nas_controller=nas_controller if nas_controller is not None else None,
        )
        if nas_space is not None:
            defaults = nas_space.defaults()
            merged_arch = defaults.copy()
            merged_arch.update(initial_architecture)
            initial_architecture = merged_arch
        self.cognition = SelfEvolvingCognition(initial_architecture, self.evolver)
        self.score_weights = self._normalise_score_weights(score_weights)
        self.recorder = recorder
        self.specialists = SpecialistModuleRegistry(specialist_modules)
        self.structural_manager = structural_manager
        self._structural_interval = (
            max(1, int(structural_interval))
            if structural_interval is not None
            else None
        )
        self._cycle_count = 0
        self.performance_diagnoser = performance_diagnoser
        self.strategy_adjuster = strategy_adjuster
        self._enforce_elite = enforce_elite
        self._regression_tolerance = float(regression_tolerance)
        self._best_architecture = self.cognition.architecture.copy()
        self._best_performance = self.evolver.fitness_fn(self._best_architecture)
        self._safety = EvolutionSafetyConfig.from_sources(safety_config) if EvolutionSafetyConfig is not None else None
        self._safety_queue = (
            ArchitectureApprovalQueue(max_pending=self._safety.max_pending_reviews)
            if self._safety is not None and ArchitectureApprovalQueue is not None
            else None
        )
        self._sandbox_runner = None
        if self._safety is not None:
            custom = None
            if isinstance(safety_config, Mapping):
                custom = safety_config.get("sandbox_runner")
                sandbox_cfg = safety_config.get("sandbox")
                if custom is None and isinstance(sandbox_cfg, Mapping):
                    custom = sandbox_cfg.get("runner")
            if callable(custom):
                self._sandbox_runner = custom
            elif PytestSandboxRunner is not None and self._safety.sandbox_pytest_paths:
                self._sandbox_runner = PytestSandboxRunner(
                    paths=self._safety.sandbox_pytest_paths,
                    timeout_s=self._safety.sandbox_timeout_s,
                    extra_args=self._safety.sandbox_pytest_args,
                )
        self._last_safety_decision: Any = None
        if self.recorder is not None and self.cognition.history:
            self.recorder.record(
                self.cognition.history[0], previous_architecture=None, annotations=None
            )

    # ------------------------------------------------------------------
    def run_evolution_cycle(
        self,
        metrics: Iterable[MetricEvent],
        benchmarks: Optional[Iterable[CognitiveBenchmarkResult]] = None,
        benchmark_weights: Optional[Dict[str, float]] = None,
        task: Optional[TaskContext] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Evolve the architecture using the provided ``metrics``.

        The metrics are aggregated into a single performance score. Candidate
        architectures are generated via the underlying genetic algorithm and
        the best one replaces the current architecture. The new architecture is
        appended to the evolution history and returned.
        """

        self._cycle_count += 1
        metrics_list = list(metrics) if metrics is not None else []
        benchmarks_list = list(benchmarks) if benchmarks is not None else []
        no_feedback = not metrics_list and not benchmarks_list
        diag_report: Optional[Dict[str, Any]] = None
        structural_due = (
            self.structural_manager is not None
            and (
                self._structural_interval is None
                or self._cycle_count % self._structural_interval == 0
            )
        )
        if no_feedback and not structural_due:
            return self.cognition.architecture

        if no_feedback:
            performance = self.evolver.fitness_fn(self.cognition.architecture)
            summary: Dict[str, float] = {
                "resource_score": 0.0,
                "cognitive_score": 0.0,
                "combined_performance": float(performance),
            }
            base_arch = self.cognition.architecture
            base_perf = performance
            summary["source"] = "baseline"
        else:
            resource_score = (
                sum(self.cognition._score_event(m) for m in metrics_list) / len(metrics_list)
                if metrics_list
                else None
            )
            cognitive_score = (
                aggregate_benchmark_score(benchmarks_list, benchmark_weights)
                if benchmarks_list
                else None
            )
            performance, summary = self._combine_scores(resource_score, cognitive_score, benchmarks_list)
            performance, diag_report = self._apply_diagnoser_feedback(
                performance, summary, metrics_list
            )
            performance, summary = self._apply_multiobjective_penalty(
                performance, summary, metrics_list
            )

            # Generate a candidate architecture from the genetic algorithm.
            base_arch = self.evolver.evolve_architecture(self.cognition.architecture, performance)
            base_perf = self.evolver.fitness_fn(base_arch)
            summary["source"] = "genetic"

        chosen_arch = base_arch
        chosen_perf = base_perf

        selected_specialist: Optional[SpecialistModule] = None
        specialist_perf: Optional[float] = None
        if task is not None:
            selected_specialist = self.specialists.select_best(task)
            if selected_specialist is not None:
                try:
                    specialist_arch = selected_specialist.solver(
                        self.cognition.architecture.copy(), task
                    )
                except Exception:
                    specialist_arch = None
                if isinstance(specialist_arch, Mapping):
                    specialist_arch = dict(specialist_arch)
                if isinstance(specialist_arch, dict):
                    specialist_perf = self.evolver.fitness_fn(specialist_arch)
                    self.specialists.update_performance(
                        selected_specialist.name, specialist_perf
                    )
                    summary["specialist_module"] = selected_specialist.name
                    summary["specialist_performance"] = float(specialist_perf)
                    if specialist_perf > chosen_perf:
                        chosen_arch = specialist_arch
                        chosen_perf = specialist_perf
                        summary["source"] = "specialist"
                else:
                    summary["specialist_module"] = selected_specialist.name
                    summary["specialist_performance"] = None

        structural_candidate = self._evaluate_structural_candidate(
            metrics_list, chosen_arch, chosen_perf, structural_due
        )
        if structural_candidate is not None:
            proposal, structural_arch, structural_perf = structural_candidate
            summary["structural_reason"] = proposal.reason
            summary["structural_score"] = float(proposal.score)
            summary["structural_performance"] = float(structural_perf)
            if structural_perf > chosen_perf:
                chosen_arch = structural_arch
                chosen_perf = structural_perf
                summary["source"] = "structural"
                if self.structural_manager is not None:
                    self.structural_manager.commit_proposal(
                        proposal,
                        performance=structural_perf,
                        extra_metrics={"from_engine": 1.0},
                    )

        if self.strategy_adjuster is not None:
            chosen_arch, chosen_perf, strategy_actions = self._apply_strategy_adjustments(
                chosen_arch, chosen_perf, diag_report
            )
            if strategy_actions:
                summary["strategy_actions"] = [action.reason for action in strategy_actions]
                summary["strategy_updates"] = {action.parameter: action.value for action in strategy_actions}

        if self._enforce_elite and chosen_perf + self._regression_tolerance < self._best_performance:
            # Reject regression, keep elite.
            chosen_arch = self._best_architecture.copy()
            chosen_perf = self._best_performance
            summary["source"] = "elite_guard"

        # Optional safety gate (performance floor / manual review / sandbox validation).
        safety_cfg = self._safety
        decision: Any = None
        annotations: Optional[Dict[str, Any]] = None
        ctx = dict(context or {})
        if safety_cfg is not None and bool(getattr(safety_cfg, "enabled", False)) and evaluate_candidate is not None:
            effective = safety_cfg
            if ctx.get("skip_manual_review") and bool(getattr(safety_cfg, "manual_review_enabled", False)):
                effective = replace(safety_cfg, manual_review_enabled=False)
            history_perfs = [float(r.performance) for r in self.cognition.history] if self.cognition.history else []
            decision = evaluate_candidate(
                config=effective,
                history_performances=history_perfs,
                current_architecture=self.cognition.architecture,
                candidate_architecture=chosen_arch,
                candidate_performance=float(chosen_perf),
                approval_queue=self._safety_queue,
                extra_details={"source": str(summary.get("source") or "evolution")},
            )
            self._last_safety_decision = decision
            if getattr(decision, "requires_review", False):
                summary["source"] = "manual_review_required"
                summary["safety_reason"] = getattr(decision, "reason", "manual_review_required")
                chosen_arch = self.cognition.architecture.copy()
                chosen_perf = self.evolver.fitness_fn(chosen_arch)
            elif not getattr(decision, "allowed", True):
                summary["source"] = "safety_rejected"
                summary["safety_reason"] = getattr(decision, "reason", "safety_rejected")
                chosen_arch = self.cognition.architecture.copy()
                chosen_perf = self.evolver.fitness_fn(chosen_arch)
            else:
                summary["safety_delta_l1"] = float(getattr(decision, "delta_l1", 0.0))
                baseline = getattr(decision, "baseline_mean", None)
                if isinstance(baseline, (int, float)):
                    summary["safety_baseline_mean"] = float(baseline)
                annotations = {
                    "safety": {
                        "decision": getattr(decision, "reason", "accepted"),
                        "delta_l1": float(getattr(decision, "delta_l1", 0.0)),
                        "baseline_mean": getattr(decision, "baseline_mean", None),
                        "baseline_window": getattr(decision, "baseline_window", None),
                        "review_id": getattr(decision, "review_id", None),
                        "details": getattr(decision, "details", None),
                    }
                }

        if (
            safety_cfg is not None
            and bool(getattr(safety_cfg, "sandbox_enabled", False))
            and self._sandbox_runner is not None
            and not ctx.get("skip_sandbox")
            and decision is not None
            and getattr(decision, "allowed", True)
        ):
            ok = bool(self._sandbox_runner())
            if not ok:
                summary["source"] = "sandbox_rejected"
                summary["safety_reason"] = "sandbox_failed"
                chosen_arch = self.cognition.architecture.copy()
                chosen_perf = self.evolver.fitness_fn(chosen_arch)
                if annotations is not None:
                    annotations.setdefault("safety", {})["sandbox"] = "failed"
            else:
                if annotations is not None:
                    annotations.setdefault("safety", {})["sandbox"] = "passed"

        # Update elite tracker only after all gates have finalised the chosen architecture.
        if chosen_perf > self._best_performance and summary.get("source") not in {
            "manual_review_required",
            "safety_rejected",
            "sandbox_rejected",
        }:
            self._best_performance = float(chosen_perf)
            self._best_architecture = chosen_arch.copy()

        self.cognition.version += 1
        self.cognition.architecture = chosen_arch
        self.cognition.history.append(
            EvolutionRecord(self.cognition.version, chosen_arch.copy(), chosen_perf, summary)
        )
        if self.recorder is not None:
            previous = None
            if len(self.cognition.history) >= 2:
                previous = self.cognition.history[-2].architecture
            self.recorder.record(
                self.cognition.history[-1],
                previous_architecture=previous,
                annotations=annotations,
            )
        return chosen_arch

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
        if (
            run_sandbox
            and self._safety is not None
            and bool(getattr(self._safety, "sandbox_enabled", False))
            and self._sandbox_runner is not None
        ):
            if not bool(self._sandbox_runner()):
                return False

        before = self.cognition.architecture.copy()
        self.cognition.version += 1
        self.cognition.architecture = dict(pending.architecture)
        if float(pending.performance) > self._best_performance:
            self._best_performance = float(pending.performance)
            self._best_architecture = dict(pending.architecture)
        self.cognition.history.append(
            EvolutionRecord(
                self.cognition.version,
                dict(pending.architecture),
                float(pending.performance),
                {"source": "manual_review_approved", "manual_review_id": pending.id},
            )
        )
        if self.recorder is not None:
            self.recorder.record(
                self.cognition.history[-1],
                previous_architecture=before,
                annotations={"safety": {"manual_review_id": pending.id, "decision": "approved"}},
            )
        self._safety_queue.pop(update_id)
        return True

    def deny_pending_architecture_update(self, update_id: str) -> bool:
        if self._safety_queue is None:
            return False
        return self._safety_queue.pop(update_id) is not None

    # ------------------------------------------------------------------
    def register_specialist_module(self, module: SpecialistModule) -> None:
        """Register a specialist module that can propose architectures."""

        self.specialists.register(module)

    # ------------------------------------------------------------------
    def register_structural_manager(self, manager: StructuralEvolutionManager) -> None:
        """Attach a structural evolution manager for topology mutations."""

        self.structural_manager = manager

    # ------------------------------------------------------------------
    def select_specialist_module(self, task: TaskContext) -> Optional[SpecialistModule]:
        """Expose registry selection primarily for testing and inspection."""

        return self.specialists.select_best(task)

    # ------------------------------------------------------------------
    def rollback(self, version: int) -> Dict[str, float]:
        """Rollback to a previous architecture version."""

        return self.cognition.rollback(version)

    # ------------------------------------------------------------------
    def history(self) -> List[EvolutionRecord]:
        """Return the evolution history."""

        return self.cognition.history

    # ------------------------------------------------------------------
    def _normalise_score_weights(self, weights: Optional[Dict[str, float]]) -> Dict[str, float]:
        base = {"resource": 0.5, "cognitive": 0.5}
        if not weights:
            return base
        base.update({k: max(0.0, float(v)) for k, v in weights.items()})
        total = sum(base.values())
        if total <= 0:
            return {"resource": 0.5, "cognitive": 0.5}
        return {k: v / total for k, v in base.items()}

    # ------------------------------------------------------------------
    def _combine_scores(
        self,
        resource_score: Optional[float],
        cognitive_score: Optional[float],
        benchmarks: List[CognitiveBenchmarkResult],
    ) -> tuple[float, Dict[str, float]]:
        weights = self.score_weights
        summaries: Dict[str, float] = {
            "resource_score": float(resource_score) if resource_score is not None else 0.0,
            "cognitive_score": float(cognitive_score) if cognitive_score is not None else 0.0,
        }

        active_weights: Dict[str, float] = {}
        if resource_score is not None:
            active_weights["resource"] = weights.get("resource", 0.0)
        if cognitive_score is not None:
            active_weights["cognitive"] = weights.get("cognitive", 0.0)
            summaries.update(summarise_benchmarks(benchmarks))

        if not active_weights:
            return 0.0, summaries

        total = sum(active_weights.values())
        if total <= 0:
            total = len(active_weights)
            active_weights = {k: 1.0 for k in active_weights}

        combined = 0.0
        if resource_score is not None:
            combined += active_weights["resource"] / total * resource_score
        if cognitive_score is not None:
            combined += active_weights["cognitive"] / total * cognitive_score

        summaries["combined_performance"] = float(combined)
        return combined, summaries

    # ------------------------------------------------------------------
    def _evaluate_structural_candidate(
        self,
        metrics: List[MetricEvent],
        base_arch: Dict[str, float],
        base_perf: float,
        structural_due: bool,
    ) -> Optional[tuple[StructuralProposal, Dict[str, float], float]]:
        manager = self.structural_manager
        if manager is None or not structural_due:
            return None
        bottlenecks = self._extract_bottlenecks(metrics)
        candidate_modules = [
            name for name, gate in getattr(manager, "module_gates", {}).items() if gate < 0.5
        ]
        proposal = manager.evolve_structure(
            performance=base_perf,
            bottlenecks=bottlenecks,
            candidate_modules=candidate_modules,
            commit=False,
        )
        structural_arch = manager.as_architecture(proposal, base_arch)
        structural_perf = self.evolver.fitness_fn(structural_arch)
        return proposal, structural_arch, structural_perf

    # ------------------------------------------------------------------
    def _extract_bottlenecks(self, metrics: List[MetricEvent]) -> List[tuple[str, float]]:
        """Identify slowest modules from metric events."""

        if not metrics:
            return []
        latencies: Dict[str, List[float]] = {}
        for event in metrics:
            module = getattr(event, "module", None)
            if module is None:
                continue
            latencies.setdefault(module, []).append(float(getattr(event, "latency", 0.0)))
        averages = [
            (module, sum(vals) / max(len(vals), 1))
            for module, vals in latencies.items()
            if vals
        ]
        averages.sort(key=lambda item: item[1], reverse=True)
        return averages

    # ------------------------------------------------------------------ #
    def _apply_diagnoser_feedback(
        self,
        performance: float,
        summary: Dict[str, float],
        metrics: List[MetricEvent],
    ) -> tuple[float, Optional[Dict[str, Any]]]:
        if self.performance_diagnoser is None or not metrics:
            return performance, None
        report = self.performance_diagnoser.diagnose(metrics, aggregate=summary)
        issues = report.get("issues", [])
        penalty = 0.05 * len(issues)
        adjusted = performance - penalty
        summary["diagnostic_penalty"] = penalty
        summary["diagnostic_issues"] = [getattr(issue, "kind", "") for issue in issues]
        return adjusted, report

    # ------------------------------------------------------------------ #
    def _apply_strategy_adjustments(
        self,
        architecture: Dict[str, float],
        performance: float,
        diag_report: Optional[Dict[str, Any]],
    ) -> tuple[Dict[str, float], float, List[Any]]:
        issues = diag_report.get("issues") if diag_report else None
        if self.strategy_adjuster is None or not issues:
            return architecture, performance, []
        proposal = self.strategy_adjuster.propose(issues, current_params=architecture)
        updates: Dict[str, float] = proposal.get("updates", {})
        actions = proposal.get("actions", [])
        if updates:
            architecture = {**architecture, **updates}
            performance = self.evolver.fitness_fn(architecture)
        return architecture, performance, actions

    # ------------------------------------------------------------------ #
    def _apply_multiobjective_penalty(
        self,
        performance: float,
        summary: Dict[str, float],
        metrics: List[MetricEvent],
    ) -> tuple[float, Dict[str, float]]:
        if not metrics:
            return performance, summary
        adjusted, details = adjust_performance(performance, metrics)
        summary.update(details)
        return adjusted, summary
