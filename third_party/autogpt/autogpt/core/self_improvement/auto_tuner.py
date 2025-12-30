"""Automatic self-improvement engine based on experience logs."""
from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from autogpt.core.configuration.learning import LearningConfiguration
from autogpt.core.learning.experience_store import ExperienceLogStore, ExperienceRecord
from autogpt.core.self_improvement.ability_tracker import AbilityScoreTracker
from autogpt.core.self_improvement.validation import PlanValidator
from autogpt.core.self_improvement.replay import ScenarioLoader
from autogpt.core.self_improvement.replay_validator import ReplayValidator


@dataclass
class CommandStats:
    name: str
    successes: int
    total: int

    @property
    def success_rate(self) -> float:
        return self.successes / self.total if self.total else 0.0


class SelfImprovementEngine:
    """Evaluates experience logs and generates improvement plans."""

    def __init__(
        self,
        config: LearningConfiguration,
        store: ExperienceLogStore,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._store = store
        self._logger = logger or logging.getLogger(__name__)
        self._state_path = Path(self._config.improvement_state_path)
        self._plan_path = Path(self._config.plan_output_path)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._plan_path.parent.mkdir(parents=True, exist_ok=True)
        self._baseline_path = Path(self._config.baseline_success_path)
        self._baseline_path.parent.mkdir(parents=True, exist_ok=True)
        self._replay_loader = ScenarioLoader(Path(self._config.replay_scenarios_dir))
        self._replay_validator = ReplayValidator(Path(self._config.replay_reports_dir), logger=self._logger)
        self._validator = PlanValidator(
            reports_dir=Path(self._config.validation_reports_dir),
            min_success_improvement=self._config.min_success_improvement,
            protected_commands=self._config.protected_commands,
            logger=self._logger,
        )

        self._ability_tracker = AbilityScoreTracker(
            history_path=Path(self._config.ability_history_path),
            logger=self._logger,
            low_score_threshold=self._config.ability_low_score_threshold,
            streak_length=self._config.ability_low_score_streak,
            history_limit=self._config.ability_history_limit,
        )
        self._last_ability_report: dict[str, Any] | None = None

    def evaluate_and_apply(
        self,
        evaluation_summary: dict[str, float] | None = None,
        benchmark_results: Any | None = None,
    ) -> dict[str, Any] | None:
        records = list(self._store)
        record_count = len(records)
        if record_count < max(self._config.min_records or 0, 1):
            return None

        ability_report: dict[str, Any] | None = None
        if evaluation_summary is not None or benchmark_results is not None:
            ability_report = self._ability_tracker.update(
                evaluation_summary=evaluation_summary,
                benchmark_results=benchmark_results,
            )
        else:
            ability_report = self._ability_tracker.latest_report()
        self._last_ability_report = ability_report

        window = self._select_window(records)
        metrics = self._compute_metrics(window)
        state = self._load_state()
        baseline = state.get("baseline_success") if state else None
        baseline_reference = self._load_baseline_reference()
        if baseline is None and baseline_reference is not None:
            baseline = baseline_reference
        current_plan = state.get("current_plan") if state else None
        previous_plan = state.get("previous_plan") if state else None

        if baseline is not None and metrics["overall_success"] < baseline - (self._config.rollback_tolerance or 0.0):
            if previous_plan:
                self._logger.warning(
                    "Rolling back self-improvement plan due to performance regression "
                    "(success %.3f -> %.3f)",
                    baseline,
                    metrics["overall_success"],
                )
                self._write_plan(previous_plan)
                self._save_state(
                    self._augment_with_ability_state(
                        {
                            "baseline_success": metrics["overall_success"],
                            "current_plan": previous_plan,
                            "previous_plan": None,
                            "last_record_count": record_count,
                        },
                        ability_report,
                    )
                )
                return previous_plan

        new_plan = self._build_plan(metrics, window, ability_report)
        if not new_plan:
            self._refresh_state(metrics, current_plan, record_count, ability_report)
            return None

        validation = self._validator.validate(
            baseline_success=baseline_reference if baseline_reference is not None else baseline,
            current_success=metrics["overall_success"],
            plan=new_plan,
            metrics=metrics,
        )
        if not validation.approved:
            self._logger.info(
                "Validation rejected plan (delta %.3f < threshold %.3f).",
                validation.details.get("delta"),
                self._config.min_success_improvement,
            )
            self._refresh_state(metrics, current_plan, record_count, ability_report)
            return None

        scenarios = self._replay_loader.load()
        replay_results = {}
        if scenarios:
            def _execute(scenario):
                # TODO: integrate real replay harness; placeholder returns True if all commands succeeded
                return all(record.result_status and record.result_status.lower() == "success" for record in scenario.records)
            replay_results = self._replay_validator.evaluate_scenarios(scenarios, _execute)
            if not all(replay_results.values()):
                self._logger.info("Replay validation failed for scenario(s): %s", [name for name, ok in replay_results.items() if not ok])
                self._refresh_state(metrics, current_plan, record_count, ability_report)
                return None
        if current_plan and self._plans_equal(current_plan, new_plan):
            self._refresh_state(metrics, current_plan, record_count, ability_report)
            return None

        if current_plan:
            state_previous = current_plan
        else:
            state_previous = None

        self._write_plan(new_plan)
        if new_plan:
            self._generate_prompt_candidate(new_plan, metrics, window)
        self._save_state(
            self._augment_with_ability_state(
                {
                    "baseline_success": metrics["overall_success"],
                    "current_plan": new_plan,
                    "previous_plan": state_previous,
                    "last_record_count": record_count,
                },
                ability_report,
            )
        )
        self._store_baseline_reference(metrics["overall_success"])
        self._logger.info(
            "Applied new self-improvement plan: success %.3f, disabled=%s, preferred=%s",
            metrics["overall_success"],
            new_plan.get("disabled_commands"),
            new_plan.get("preferred_commands"),
        )
        return new_plan

    def _refresh_state(
        self,
        metrics: dict[str, Any],
        current_plan: dict[str, Any] | None,
        record_count: int,
        ability_report: dict[str, Any] | None,
    ) -> None:
        state = self._load_state() or {}
        state.update(
            self._augment_with_ability_state(
                {
                    "baseline_success": max(
                        metrics["overall_success"], state.get("baseline_success", 0.0)
                    ),
                    "current_plan": current_plan,
                    "last_record_count": record_count,
                },
                ability_report,
            )
        )
        self._save_state(state)

    def _select_window(self, records: list[ExperienceRecord]) -> list[ExperienceRecord]:
        window_size = max(self._config.min_records or 0, (self._config.batch_size or 1) * 2)
        return records[-window_size:]

    def _compute_metrics(self, records: Iterable[ExperienceRecord]) -> dict[str, Any]:
        total = 0
        successes = 0
        command_stats: dict[str, CommandStats] = {}
        for record in records:
            total += 1
            status = (record.result_status or "").lower()
            is_success = status == "success"
            if is_success:
                successes += 1
            name = record.command_name or ""
            if name not in command_stats:
                command_stats[name] = CommandStats(name=name, successes=0, total=0)
            entry = command_stats[name]
            entry.total += 1
            if is_success:
                entry.successes += 1
        overall_success = successes / total if total else 0.0
        return {
            "overall_success": overall_success,
            "command_stats": command_stats,
            "total": total,
        }

    def _build_plan(
        self,
        metrics: dict[str, Any],
        window: Iterable[ExperienceRecord],
        ability_report: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        stats = metrics["command_stats"]
        disabled: list[str] = []
        preferred: list[str] = []
        hints: list[str] = []
        for command in sorted(
            stats.values(), key=lambda c: c.success_rate, reverse=True
        ):
            if command.total < 3:
                continue
            rate = command.success_rate
            if rate >= 0.8:
                preferred.append(command.name)
            elif rate <= 0.2:
                disabled.append(command.name)
        top_success = [c for c in stats.values() if c.total >= 3]
        top_success.sort(key=lambda c: c.success_rate, reverse=True)
        if top_success:
            best = top_success[0]
            hints.append(
                f"Command '{best.name}' is succeeding {best.successes}/{best.total} times; "
                "consider prioritising it early in plans."
            )
        failing = [c for c in stats.values() if c.total >= 3 and c.success_rate <= 0.3]
        failing.sort(key=lambda c: c.success_rate)
        if failing:
            worst = failing[0]
            hints.append(
                f"Command '{worst.name}' struggles ({worst.successes}/{worst.total}). "
                "Avoid unless necessary or gather more context first."
            )

        targeted_actions = self._suggest_targeted_actions(ability_report)
        include_plan = bool(disabled or preferred or hints or targeted_actions)
        if not include_plan:
            return None
        plan = {
            "generated_at": datetime.utcnow().isoformat(),
            "overall_success": metrics["overall_success"],
            "disabled_commands": disabled,
            "preferred_commands": preferred,
            "prompt_hints": "\n".join(hints) if hints else "",
            "window_size": metrics["total"],
        }
        if ability_report:
            plan["ability_scores"] = ability_report.get("scores", {})
        if targeted_actions:
            plan["targeted_actions"] = targeted_actions
        return plan

    def _generate_prompt_candidate(
        self, plan: dict[str, Any], metrics: dict[str, Any], window: Iterable[ExperienceRecord]
    ) -> None:
        if not self._config.generate_prompt_candidates:
            return
        try:
            candidates_dir = Path(self._config.prompt_candidates_dir)
        except Exception:
            self._logger.exception("Invalid prompt_candidates_dir")
            return
        try:
            candidates_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            self._logger.exception("Failed to create prompt candidates directory")
            return

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        candidate_path = candidates_dir / f"candidate_{timestamp}.md"

        preferred = plan.get("preferred_commands") or []
        disabled = plan.get("disabled_commands") or []
        hints = plan.get("prompt_hints") or ""
        overall = f"{metrics.get('overall_success', 0.0):.2%}"

        top_examples = []
        for record in window:
            if record.result_status and record.result_status.lower() == "success":
                summary = record.result_summary or ""
                if summary:
                    top_examples.append(summary)
            if len(top_examples) >= 3:
                break

        sections = [
            f"# Prompt Candidate {timestamp}",
            "",
            "## Context",
            f"- Window size: {metrics.get('total')} records",
            f"- Overall success rate: {overall}",
            f"- Preferred commands: {', '.join(preferred) if preferred else 'None'}",
            f"- Disabled commands: {', '.join(disabled) if disabled else 'None'}",
            "",
            "## Strategy Hints",
            hints or "_No additional hints generated._",
            "",
            "## Suggested System Prompt",
            "You are AutoGPT executing tasks autonomously. Focus on the commands with the highest success rate and avoid those repeatedly failing. Always explain your reasoning briefly before acting, and verify the outcome of each command before moving on. Use the following priorities:",
            f"1. Prefer commands: {', '.join(preferred) if preferred else 'No specific preference'}",
            f"2. Avoid commands: {', '.join(disabled) if disabled else 'None'}",
            "3. When a command fails twice, fall back to an alternative approach or seek additional context.",
        ]

        if top_examples:
            sections.extend(
                [
                    "",
                    "## Recent Successful Outputs",
                    *[f"- {example}" for example in top_examples],
                ]
            )

        try:
            candidate_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
        except Exception:
            self._logger.exception("Failed to write prompt candidate")
            return

        self._prune_candidates(candidates_dir)

    def _prune_candidates(self, candidates_dir: Path) -> None:
        max_candidates = self._config.max_prompt_candidates or 0
        if max_candidates <= 0:
            return
        try:
            files = sorted(
                [p for p in candidates_dir.glob("candidate_*.md") if p.is_file()],
                key=lambda p: p.stat().st_mtime,
            )
        except Exception:
            self._logger.exception("Failed to inspect prompt candidates for pruning")
            return
        while len(files) > max_candidates:
            path = files.pop(0)
            with contextlib.suppress(Exception):
                path.unlink()

    def _load_baseline_reference(self) -> float | None:
        if not self._baseline_path.exists():
            return None
        try:
            data = json.loads(self._baseline_path.read_text(encoding="utf-8"))
            return float(data.get("baseline_success"))
        except Exception:
            self._logger.exception("Failed to load baseline reference")
            return None

    def _store_baseline_reference(self, success: float) -> None:
        try:
            self._baseline_path.write_text(
                json.dumps({"baseline_success": success}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            self._logger.exception("Failed to store baseline reference")

    def _load_state(self) -> dict[str, Any]:
        if not self._state_path.exists():
            return {}
        try:
            return json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            self._logger.exception("Failed to load self-improvement state")
            return {}

    def _save_state(self, state: dict[str, Any]) -> None:
        try:
            self._state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            self._logger.exception("Failed to write self-improvement state")

    def _write_plan(self, plan: dict[str, Any]) -> None:
        try:
            self._plan_path.write_text(
                json.dumps(plan, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            self._logger.exception("Failed to write improvement plan")

    @staticmethod
    def _plans_equal(plan_a: dict[str, Any], plan_b: dict[str, Any]) -> bool:
        def canonical(plan: dict[str, Any]) -> dict[str, Any]:
            return {
                "disabled": sorted(plan.get("disabled_commands") or []),
                "preferred": sorted(plan.get("preferred_commands") or []),
                "hints": plan.get("prompt_hints", ""),
            }

        return canonical(plan_a) == canonical(plan_b)

    def _augment_with_ability_state(
        self, state: dict[str, Any], ability_report: dict[str, Any] | None
    ) -> dict[str, Any]:
        if not ability_report:
            return state
        ability_state = {
            "latest_scores": ability_report.get("scores", {}),
            "weak_abilities": ability_report.get("weak_abilities", []),
            "history": ability_report.get("history", []),
        }
        state = dict(state)
        state["ability_state"] = ability_state
        return state

    def _suggest_targeted_actions(
        self, ability_report: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        if not ability_report:
            return []
        weak = ability_report.get("weak_abilities") or []
        if not weak:
            return []
        actions: list[dict[str, Any]] = []
        for entry in weak:
            name = entry.get("name") or entry.get("ability")
            if not name:
                continue
            actions.append(
                {
                    "ability": name,
                    "streak": entry.get("streak"),
                    "latest_score": entry.get("latest_score"),
                    "recommendation": self._recommendation_for_ability(name),
                }
            )
        return actions

    def _recommendation_for_ability(self, ability: str) -> str:
        ability = ability.lower()
        if ability in {"logic_reasoning", "general_problem_solving"}:
            return (
                "Boost analytical command weights and schedule reasoning-focused"
                " replay scenarios."
            )
        if ability in {"creativity"}:
            return (
                "Trigger knowledge augmentation for creative domains and request"
                " fresh brainstorming datasets."
            )
        if ability in {"planning"}:
            return (
                "Prioritise planning abilities, expand situational context gathering,"
                " and rehearse multi-step scenarios."
            )
        if ability in {"knowledge"}:
            return (
                "Initiate retrieval tuning and ingest recent knowledge base updates"
                " to close factual gaps."
            )
        if ability in {"execution_efficiency"}:
            return (
                "Optimise high-latency commands, enable caching, and consider"
                " lightweight tool alternatives."
            )
        if ability in {"fairness_alignment"}:
            return (
                "Run bias audits on recent data and rebalance decision heuristics"
                " for sensitive groups."
            )
        if ability in {"tool_usage"}:
            return (
                "Review tool invocation strategies and refresh API credentials to"
                " reduce integration failures."
            )
        if ability in {"multimodal"}:
            return (
                "Schedule multimodal fine-tuning passes and verify perception"
                " pipelines with targeted fixtures."
            )
        return "Queue a focused learning task targeting this capability and re-run validation."
