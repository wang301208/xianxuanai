"""Agent-layer self-improvement controller built on the evolution stack.

This module provides a lightweight, *online* self-improvement loop that turns
recent execution telemetry (MetricEvent streams) into updates for a small
"strategy genome". The genome is meant to represent *agent-level* knobs such as
planner/policy flags and prompt-strategy parameters.

Design goals:
- Deterministic & test-friendly by default (seeded GA; no LLM required)
- Optional persistence (JSON) so improvement can accumulate across runs
- Reuses existing components: PerformanceDiagnoser + StrategyAdjuster + GA
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - support both repo-root and `modules/` on sys.path
    from modules.monitoring.collector import MetricEvent
    from modules.monitoring.performance_diagnoser import DiagnosticIssue, PerformanceDiagnoser
except ModuleNotFoundError:  # pragma: no cover
    from monitoring.collector import MetricEvent  # type: ignore
    from monitoring.performance_diagnoser import DiagnosticIssue, PerformanceDiagnoser  # type: ignore

from .evolving_cognitive_architecture import GAConfig, GeneticAlgorithm
from .strategy_adjuster import StrategyAdjuster

try:  # pragma: no cover - optional meta-NAS controller
    from .meta_nas import MetaNASController
except Exception:  # pragma: no cover
    MetaNASController = None  # type: ignore[assignment]


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_int_flag(value: float) -> float:
    return 1.0 if float(value) >= 0.5 else 0.0


def _default_genome() -> Dict[str, float]:
    # Keep defaults aligned with modules/brain/whole_brain.py `_init_self_evolution`.
    return {
        "policy_learning_rate": 0.08,
        "policy_exploration_rate": 0.12,
        "planner_structured_flag": 1.0,
        "planner_reinforcement_flag": 0.0,
        # LLM prompt strategy genes (agent-facing).
        "llm_prompt_variant": 0.0,
        "llm_prompt_json_strictness": 0.6,
        "llm_prompt_safety_bias": 0.8,
        # Knowledge acquisition knobs (toolchain-facing).
        "knowledge_acq_enabled_flag": 0.0,
        "knowledge_acq_web_flag": 0.0,
        "knowledge_acq_top_k": 5.0,
        "knowledge_acq_max_files": 800.0,
        "knowledge_acq_embedding_dim": 128.0,
    }


@dataclass
class StrategyGenome:
    """Mutable strategy genome updated over time."""

    genes: Dict[str, float] = field(default_factory=_default_genome)
    version: int = 0
    updated_at: float = field(default_factory=time.time)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> Dict[str, float]:
        return dict(self.genes)


@dataclass(frozen=True)
class SelfImprovementUpdate:
    """Result of a single improvement step."""

    version: int
    genes: Dict[str, float]
    diagnostics: Dict[str, Any]
    strategy: Dict[str, Any]
    source: str


class AgentSelfImprovementController:
    """Observe telemetry and evolve agent strategy parameters."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        genome: StrategyGenome | None = None,
        diagnoser: PerformanceDiagnoser | None = None,
        adjuster: StrategyAdjuster | None = None,
        enable_ga: bool = True,
        enable_meta_nas: bool = True,
        meta_nas: Any | None = None,
        ga_config: GAConfig | None = None,
        ga_seed: int | None = 0,
        persist_path: Path | str | None = None,
        min_events: int = 1,
    ) -> None:
        self.enabled = bool(enabled)
        self.genome = genome or StrategyGenome()
        self.diagnoser = diagnoser or PerformanceDiagnoser()
        self.adjuster = adjuster or StrategyAdjuster()
        self.enable_ga = bool(enable_ga)
        self.enable_meta_nas = bool(enable_meta_nas)
        self.min_events = max(0, int(min_events))

        self._persist_path = Path(persist_path) if persist_path is not None else None
        self._ga_config = ga_config or GAConfig(population_size=16, generations=4, mutation_rate=0.35, mutation_sigma=0.12)
        self._ga_seed = ga_seed
        self._meta_nas = meta_nas

        if self._persist_path is not None:
            self._load_persisted()

    # ------------------------------------------------------------------ #
    def strategy_context(self) -> Dict[str, Any]:
        """Return a compact prompt/planner-friendly view of the current genome."""

        genes = self.genome.snapshot()
        return {
            "version": int(self.genome.version),
            "genes": genes,
            "planner": {
                "structured": bool(genes.get("planner_structured_flag", 1.0) >= 0.5),
                "reinforcement": bool(genes.get("planner_reinforcement_flag", 0.0) >= 0.5),
            },
            "policy": {
                "learning_rate": float(genes.get("policy_learning_rate", 0.08)),
                "exploration_rate": float(genes.get("policy_exploration_rate", 0.12)),
            },
            "prompt": {
                "variant": int(round(float(genes.get("llm_prompt_variant", 0.0)))),
                "json_strictness": float(genes.get("llm_prompt_json_strictness", 0.6)),
                "safety_bias": float(genes.get("llm_prompt_safety_bias", 0.8)),
            },
            "knowledge_acquisition": {
                "enabled": bool(genes.get("knowledge_acq_enabled_flag", 0.0) >= 0.5),
                "web": bool(genes.get("knowledge_acq_web_flag", 0.0) >= 0.5),
                "top_k": int(round(float(genes.get("knowledge_acq_top_k", 5.0)))),
                "max_files": int(round(float(genes.get("knowledge_acq_max_files", 800.0)))),
                "embedding_dimensions": int(round(float(genes.get("knowledge_acq_embedding_dim", 128.0)))),
            },
        }

    def as_architecture_update(self) -> Dict[str, float]:
        """Return the current genome as an architecture update payload.

        This is suitable for :meth:`SelfEvolvingAIArchitecture.update_architecture`
        or any compatible interface that accepts a dict of float parameters.
        """

        return self.genome.snapshot()

    def apply_to_architecture(
        self,
        architecture: Any,
        *,
        performance: float | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> bool:
        """Apply the current genome to an attached evolving architecture.

        Returns ``True`` when an update call was attempted successfully.
        """

        update_fn = getattr(architecture, "update_architecture", None)
        if not callable(update_fn):
            return False

        metrics_payload: Dict[str, float] | None = None
        if isinstance(metrics, Mapping):
            numeric: Dict[str, float] = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    numeric[str(key)] = float(value)
            metrics_payload = numeric

        try:
            result = update_fn(self.as_architecture_update(), performance=performance, metrics=metrics_payload)
            if isinstance(result, bool):
                return bool(result)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    def observe(
        self,
        events: Sequence[MetricEvent],
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> SelfImprovementUpdate | None:
        """Update the genome from a batch of MetricEvents.

        Parameters
        ----------
        events:
            Recent telemetry events (e.g., tool steps) to analyse.
        extra:
            Optional extra context (e.g., goal text, aggregate reward, parse failures).
        """

        if not self.enabled:
            return None

        events_list = list(events or [])
        if self.min_events and len(events_list) < self.min_events:
            return None

        diagnostics = self.diagnoser.diagnose(events_list)
        issues_raw = diagnostics.get("issues")
        issues: List[DiagnosticIssue] = (
            list(issues_raw) if isinstance(issues_raw, list) else []
        )

        current = self.genome.snapshot()
        proposal = self.adjuster.propose(issues, current_params=current)
        updates = proposal.get("updates") if isinstance(proposal, dict) else {}
        updates_dict = dict(updates) if isinstance(updates, dict) else {}

        extra_payload = dict(extra or {})
        parse_failures = _safe_float(extra_payload.get("plan_parse_failures"), 0.0)
        status_hint = str(extra_payload.get("status") or "").strip().lower()
        if parse_failures > 0:
            updates_dict["llm_prompt_json_strictness"] = min(
                1.0, _safe_float(updates_dict.get("llm_prompt_json_strictness"), current.get("llm_prompt_json_strictness", 0.6)) + 0.1
            )
            updates_dict["llm_prompt_variant"] = max(
                int(round(_safe_float(updates_dict.get("llm_prompt_variant"), current.get("llm_prompt_variant", 0.0)))),
                1,
            )
        if status_hint == "blocked":
            updates_dict["llm_prompt_safety_bias"] = min(
                1.0, _safe_float(updates_dict.get("llm_prompt_safety_bias"), current.get("llm_prompt_safety_bias", 0.8)) + 0.1
            )
            updates_dict["llm_prompt_variant"] = max(
                int(round(_safe_float(updates_dict.get("llm_prompt_variant"), current.get("llm_prompt_variant", 0.0)))),
                2,
            )

        issue_kinds = {str(getattr(issue, "kind", "") or "") for issue in issues}
        if issue_kinds.intersection({"low_success_rate", "global_low_success_rate"}):
            updates_dict["knowledge_acq_enabled_flag"] = 1.0
        if issue_kinds.intersection({"high_latency", "global_high_latency"}):
            updates_dict["knowledge_acq_web_flag"] = 0.0

        merged = dict(current)
        merged.update({k: float(v) for k, v in updates_dict.items() if isinstance(v, (int, float))})
        merged = self._normalise_genome(merged)

        # Optional NAS refinement: use a bandit-controlled operator library to
        # explore more diverse candidates than pure Gaussian mutation.
        source = "adjuster"
        if self.enable_meta_nas and issues and MetaNASController is not None:
            best_candidate = self._evolve_with_meta_nas(merged, issues, extra=extra_payload)
            if isinstance(best_candidate, dict):
                merged = dict(best_candidate)
                source = "adjuster+meta_nas"
        elif self.enable_ga and issues:
            best_candidate = self._evolve_with_ga(merged, issues, extra=extra_payload)
            if isinstance(best_candidate, dict):
                merged = dict(best_candidate)
                source = "adjuster+ga"
        self.genome.version += 1
        self.genome.updated_at = time.time()
        self.genome.genes = merged

        record = {
            "ts": self.genome.updated_at,
            "version": int(self.genome.version),
            "issues": [self._issue_to_dict(issue) for issue in issues],
            "updates": dict(updates_dict),
            "source": source,
            "extra": dict(extra_payload),
        }
        if source == "adjuster+meta_nas":
            controller = self._resolve_meta_nas_controller()
            if controller is not None and hasattr(controller, "bandit"):
                try:
                    record["meta_nas"] = {"bandit": controller.bandit.snapshot()}
                except Exception:
                    pass
        self.genome.history.append(record)
        self._persist()

        return SelfImprovementUpdate(
            version=int(self.genome.version),
            genes=dict(merged),
            diagnostics=dict(diagnostics),
            strategy=dict(proposal) if isinstance(proposal, dict) else {"updates": {}, "actions": []},
            source=source,
        )

    # ------------------------------------------------------------------ #
    def _issue_to_dict(self, issue: DiagnosticIssue) -> Dict[str, Any]:
        try:
            return {
                "kind": str(issue.kind),
                "metric": str(issue.metric),
                "value": float(issue.value),
                "threshold": float(issue.threshold),
                "module": issue.module,
            }
        except Exception:
            return {"kind": getattr(issue, "kind", "unknown")}

    def _evolve_with_ga(
        self,
        current: Dict[str, float],
        issues: Sequence[DiagnosticIssue],
        *,
        extra: Mapping[str, Any] | None,
    ) -> Dict[str, float] | None:
        def _fitness(candidate: Dict[str, float]) -> float:
            return self._heuristic_fitness(candidate, issues, extra=extra)

        def _post_mutation(candidate: Dict[str, float]) -> None:
            normalised = self._normalise_genome(candidate)
            candidate.clear()
            candidate.update(normalised)

        ga = GeneticAlgorithm(_fitness, config=self._ga_config, seed=self._ga_seed, post_mutation=_post_mutation)
        best, _best_score, _history = ga.evolve(dict(current))
        best = self._normalise_genome(best)
        return best

    def _resolve_meta_nas_controller(self) -> Any | None:
        existing = getattr(self, "_meta_nas", None)
        if existing is not None:
            return existing
        if MetaNASController is None:
            return None

        def _postprocess(candidate: Dict[str, float]) -> None:
            normalised = self._normalise_genome(candidate)
            candidate.clear()
            candidate.update(normalised)

        try:
            controller = MetaNASController(
                seed=self._ga_seed,
                population_size=self._ga_config.population_size,
                generations=self._ga_config.generations,
                postprocess=_postprocess,
                reward_baseline="best",
            )
        except Exception:
            return None
        self._meta_nas = controller
        return controller

    def _evolve_with_meta_nas(
        self,
        current: Dict[str, float],
        issues: Sequence[DiagnosticIssue],
        *,
        extra: Mapping[str, Any] | None,
    ) -> Dict[str, float] | None:
        controller = self._resolve_meta_nas_controller()
        if controller is None:
            return None

        def _fitness(candidate: Dict[str, float]) -> float:
            return self._heuristic_fitness(candidate, issues, extra=extra)

        try:
            from .mutation_operators import MutationContext

            ctx = MutationContext(issues=list(issues), extra=dict(extra or {}), score_hint=None)
        except Exception:
            ctx = None

        best, _best_score, _history = controller.search(dict(current), _fitness, context=ctx)
        best = self._normalise_genome(best)
        return best

    def _heuristic_fitness(
        self,
        candidate: Dict[str, float],
        issues: Sequence[DiagnosticIssue],
        *,
        extra: Mapping[str, Any] | None,
    ) -> float:
        """Heuristic proxy for selecting a promising strategy genome.

        This avoids running full rollouts for each GA candidate. The objective
        is to move candidate parameters in directions that are known to help
        with the *currently observed* issues.
        """

        lr = _safe_float(candidate.get("policy_learning_rate"), 0.08)
        explore = _safe_float(candidate.get("policy_exploration_rate"), 0.12)
        structured = 1.0 if _safe_float(candidate.get("planner_structured_flag"), 1.0) >= 0.5 else 0.0

        prompt_json = _safe_float(candidate.get("llm_prompt_json_strictness"), 0.6)
        prompt_safety = _safe_float(candidate.get("llm_prompt_safety_bias"), 0.8)
        knowledge_acq = 1.0 if _safe_float(candidate.get("knowledge_acq_enabled_flag"), 0.0) >= 0.5 else 0.0
        knowledge_web = 1.0 if _safe_float(candidate.get("knowledge_acq_web_flag"), 0.0) >= 0.5 else 0.0

        score = 0.0
        for issue in issues:
            kind = str(getattr(issue, "kind", "") or "")
            if kind in {"low_success_rate", "global_low_success_rate"}:
                score += 1.0 * explore
                score += 0.25 * lr
                score += 0.15 * prompt_safety
                score += 0.35 * knowledge_acq
            if kind in {"high_latency", "global_high_latency"}:
                score += 0.6 * structured
                score -= 0.15 * explore
                score -= 0.2 * knowledge_acq
                score -= 0.35 * knowledge_web
            if kind in {"low_throughput", "global_low_throughput"}:
                score += 0.35 * lr
            if kind in {"high_energy", "global_high_energy"}:
                score -= 0.25 * lr

        extra_payload = dict(extra or {})
        parse_failures = _safe_float(extra_payload.get("plan_parse_failures"), 0.0)
        if parse_failures > 0:
            score += 0.4 * prompt_json
            score += 0.2 * structured
            score += 0.15 * (1.0 if int(round(_safe_float(candidate.get("llm_prompt_variant"), 0.0))) >= 1 else 0.0)

        status_hint = str(extra_payload.get("status") or "").strip().lower()
        if status_hint == "blocked":
            score += 0.25 * prompt_safety
            score += 0.2 * (1.0 if int(round(_safe_float(candidate.get("llm_prompt_variant"), 0.0))) >= 2 else 0.0)

        # Regularize: avoid drifting to extreme values without evidence.
        score -= 0.02 * abs(lr - 0.08)
        score -= 0.02 * abs(explore - 0.12)
        score -= 0.02 * abs(prompt_json - 0.6)
        score -= 0.02 * abs(prompt_safety - 0.8)
        score -= 0.02 * min(2.0, float(int(round(_safe_float(candidate.get("llm_prompt_variant"), 0.0)))))
        score -= 0.03 * knowledge_acq
        score -= 0.05 * knowledge_web

        return float(score)

    def _normalise_genome(self, genome: Dict[str, float]) -> Dict[str, float]:
        normalised = dict(genome)

        # Policy knobs.
        normalised["policy_learning_rate"] = _clamp(
            _safe_float(normalised.get("policy_learning_rate"), 0.08), 1e-4, 1.0
        )
        normalised["policy_exploration_rate"] = _clamp(
            _safe_float(normalised.get("policy_exploration_rate"), 0.12), 0.0, 1.0
        )

        # Planner flags.
        normalised["planner_structured_flag"] = _as_int_flag(
            _safe_float(normalised.get("planner_structured_flag"), 1.0)
        )
        normalised["planner_reinforcement_flag"] = _as_int_flag(
            _safe_float(normalised.get("planner_reinforcement_flag"), 0.0)
        )

        # Prompt strategy.
        variant = int(round(_safe_float(normalised.get("llm_prompt_variant"), 0.0)))
        variant = max(0, min(2, variant))
        normalised["llm_prompt_variant"] = float(variant)
        normalised["llm_prompt_json_strictness"] = _clamp(
            _safe_float(normalised.get("llm_prompt_json_strictness"), 0.6), 0.0, 1.0
        )
        normalised["llm_prompt_safety_bias"] = _clamp(
            _safe_float(normalised.get("llm_prompt_safety_bias"), 0.8), 0.0, 1.0
        )

        # Knowledge acquisition (toolchain) knobs.
        normalised["knowledge_acq_enabled_flag"] = _as_int_flag(
            _safe_float(normalised.get("knowledge_acq_enabled_flag"), 0.0)
        )
        normalised["knowledge_acq_web_flag"] = _as_int_flag(
            _safe_float(normalised.get("knowledge_acq_web_flag"), 0.0)
        )
        top_k = int(round(_safe_float(normalised.get("knowledge_acq_top_k"), 5.0)))
        top_k = max(1, min(25, top_k))
        normalised["knowledge_acq_top_k"] = float(top_k)
        max_files = int(round(_safe_float(normalised.get("knowledge_acq_max_files"), 800.0)))
        max_files = max(50, min(5000, max_files))
        normalised["knowledge_acq_max_files"] = float(max_files)
        dim = int(round(_safe_float(normalised.get("knowledge_acq_embedding_dim"), 128.0)))
        dim = max(8, min(2048, dim))
        normalised["knowledge_acq_embedding_dim"] = float(dim)

        return normalised

    def _load_persisted(self) -> None:
        path = self._persist_path
        if path is None:
            return
        try:
            if not path.exists():
                return
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return

        genes = payload.get("genes") if isinstance(payload, dict) else None
        version = payload.get("version") if isinstance(payload, dict) else None
        history = payload.get("history") if isinstance(payload, dict) else None

        if isinstance(genes, dict):
            merged = self.genome.snapshot()
            for k, v in genes.items():
                if isinstance(v, (int, float)):
                    merged[str(k)] = float(v)
            self.genome.genes = self._normalise_genome(merged)
        if isinstance(version, int):
            self.genome.version = int(version)
        if isinstance(history, list):
            self.genome.history = list(history)

    def _persist(self) -> None:
        path = self._persist_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": int(self.genome.version),
                "updated_at": float(self.genome.updated_at),
                "genes": dict(self.genome.snapshot()),
                "history": list(self.genome.history[-200:]),
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            return


__all__ = ["AgentSelfImprovementController", "SelfImprovementUpdate", "StrategyGenome"]
