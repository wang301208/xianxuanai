"""Translate diagnostic findings into actionable strategy tweaks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

try:  # pragma: no cover - support both repo-root and `modules/` on sys.path
    from modules.monitoring.performance_diagnoser import DiagnosticIssue
except ModuleNotFoundError:  # pragma: no cover
    from monitoring.performance_diagnoser import DiagnosticIssue


@dataclass
class StrategyAction:
    """Represents a proposed adjustment."""

    parameter: str
    value: float
    reason: str


class StrategyAdjuster:
    """Map performance diagnoses to hyper-parameter or mode adjustments."""

    def __init__(
        self,
        *,
        lr_bounds: tuple[float, float] = (1e-4, 1.0),
        exploration_bounds: tuple[float, float] = (0.0, 1.0),
        lr_step: float = 0.1,
        exploration_step: float = 0.05,
    ) -> None:
        self.lr_bounds = lr_bounds
        self.exploration_bounds = exploration_bounds
        self.lr_step = lr_step
        self.exploration_step = exploration_step

    # ------------------------------------------------------------------ #
    def propose(
        self,
        issues: Iterable[DiagnosticIssue],
        current_params: Optional[Dict[str, float]] = None,
    ) -> Dict[str, object]:
        """Return parameter updates and action log derived from issues."""

        params = dict(current_params or {})
        actions: List[StrategyAction] = []

        for issue in issues:
            if issue.kind in {"low_success_rate", "global_low_success_rate"}:
                self._increase_exploration(params, actions, issue)
            if issue.kind in {"high_latency", "global_high_latency"}:
                self._reduce_batch_or_enable_structure(params, actions, issue)
            if issue.kind in {"low_throughput", "global_low_throughput"}:
                self._increase_learning_rate(params, actions, issue)
            if issue.kind in {"high_energy", "global_high_energy"}:
                self._decrease_learning_rate(params, actions, issue)

        return {
            "updates": params,
            "actions": actions,
        }

    # ------------------------------------------------------------------ #
    def _increase_exploration(
        self, params: Dict[str, float], actions: List[StrategyAction], issue: DiagnosticIssue
    ) -> None:
        key = "policy_exploration_rate"
        value = params.get(key, 0.1)
        value = min(self.exploration_bounds[1], value + self.exploration_step)
        params[key] = value
        actions.append(
            StrategyAction(
                parameter=key,
                value=value,
                reason=issue.kind,
            )
        )

    # ------------------------------------------------------------------ #
    def _reduce_batch_or_enable_structure(
        self, params: Dict[str, float], actions: List[StrategyAction], issue: DiagnosticIssue
    ) -> None:
        batch_key = "memory_summary_batch_size"
        if batch_key in params:
            batch = max(1.0, params.get(batch_key, 5.0) * 0.8)
            params[batch_key] = batch
            actions.append(
                StrategyAction(
                    parameter=batch_key,
                    value=batch,
                    reason=issue.kind,
                )
            )
        struct_key = "planner_structured_flag"
        params[struct_key] = 1.0
        actions.append(
            StrategyAction(
                parameter=struct_key,
                value=1.0,
                reason=issue.kind,
            )
        )

    # ------------------------------------------------------------------ #
    def _increase_learning_rate(
        self, params: Dict[str, float], actions: List[StrategyAction], issue: DiagnosticIssue
    ) -> None:
        key = "policy_learning_rate"
        value = params.get(key, 0.05)
        value = min(self.lr_bounds[1], value + self.lr_step * value)
        params[key] = value
        actions.append(
            StrategyAction(
                parameter=key,
                value=value,
                reason=issue.kind,
            )
        )

    # ------------------------------------------------------------------ #
    def _decrease_learning_rate(
        self, params: Dict[str, float], actions: List[StrategyAction], issue: DiagnosticIssue
    ) -> None:
        key = "policy_learning_rate"
        value = params.get(key, 0.05)
        value = max(self.lr_bounds[0], value - self.lr_step * value)
        params[key] = value
        actions.append(
            StrategyAction(
                parameter=key,
                value=value,
                reason=issue.kind,
            )
        )


__all__ = ["StrategyAdjuster", "StrategyAction"]
