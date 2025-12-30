from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional

from ..memory.long_term import LongTermMemory
from ..reflection import ReflectionModule, ReflectionResult

AdjustStrategy = Callable[["StepReport", ReflectionResult], Dict[str, Any]]
RecoveryHook = Callable[["StepReport", ReflectionResult], None]


@dataclass
class StepReport:
    """Describe the outcome of a single decision or action."""

    action: str
    observation: str
    status: str = "unknown"
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> str:
        """Convert the step results into a compact summary for reflection."""

        metrics_desc = ", ".join(
            f"{key}={value:.4f}" if isinstance(value, (int, float)) else f"{key}={value}"
            for key, value in sorted(self.metrics.items())
        )
        parts = [
            f"action={self.action}",
            f"status={self.status}",
            f"observation={self.observation}",
        ]
        if metrics_desc:
            parts.append(f"metrics={metrics_desc}")
        if self.error:
            parts.append(f"error={self.error}")
        if self.retries:
            parts.append(f"retries={self.retries}")
        if self.metadata:
            serialized = json.dumps(self.metadata, sort_keys=True, ensure_ascii=True, default=str)
            parts.append(f"metadata={serialized}")
        return "; ".join(parts)


@dataclass
class RecoveryDecision:
    """Result of a self-monitoring assessment."""

    should_retry: bool
    evaluation: ReflectionResult
    revision: str
    adjustments: Dict[str, Any] | None = None
    assistance_invoked: bool = False


class SelfMonitoringSystem:
    """Closed-loop self-monitoring built on top of :class:`ReflectionModule`."""

    def __init__(
        self,
        *,
        reflection: Optional[ReflectionModule] = None,
        memory: Optional[LongTermMemory] = None,
        quality_threshold: float = 0.6,
        max_retries: int = 2,
        adjust_strategy: Optional[AdjustStrategy] = None,
        recovery_hook: Optional[RecoveryHook] = None,
    ) -> None:
        self.reflection = reflection or ReflectionModule(max_passes=2, quality_threshold=quality_threshold)
        self.memory = memory
        self.quality_threshold = quality_threshold
        self.max_retries = max_retries
        self.adjust_strategy = adjust_strategy
        self.recovery_hook = recovery_hook
        self._log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    def assess_step(self, report: StepReport) -> RecoveryDecision:
        """Evaluate ``report`` and decide whether recovery is required."""

        summary = report.to_summary()
        evaluation, revised = self.reflection.reflect(summary)
        should_retry = evaluation.confidence < self.quality_threshold and report.retries < self.max_retries
        adjustments: Dict[str, Any] | None = None
        assistance_invoked = False

        if should_retry:
            if self.adjust_strategy:
                try:
                    adjustments = self.adjust_strategy(report, evaluation)
                except Exception:  # pragma: no cover - defensive
                    adjustments = None
            else:
                adjustments = self._default_adjustments(report, evaluation)

        if should_retry and self.recovery_hook:
            try:
                self.recovery_hook(report, evaluation)
                assistance_invoked = True
            except Exception:  # pragma: no cover - defensive
                assistance_invoked = False

        record = {
            "action": report.action,
            "status": report.status,
            "summary": summary,
            "evaluation": evaluation.__dict__,
            "revision": revised,
            "should_retry": should_retry,
            "adjustments": adjustments,
            "retries": report.retries,
        }
        self._log.append(record)
        self._persist(record)

        return RecoveryDecision(
            should_retry=should_retry,
            evaluation=evaluation,
            revision=revised,
            adjustments=adjustments,
            assistance_invoked=assistance_invoked,
        )

    # ------------------------------------------------------------------
    def review_sequence(self, reports: Iterable[StepReport]) -> list[RecoveryDecision]:
        """Assess multiple reports in order and return the corresponding decisions."""

        decisions: list[RecoveryDecision] = []
        for report in reports:
            decisions.append(self.assess_step(report))
        return decisions

    def history(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the monitoring records."""

        return list(self._log)

    # ------------------------------------------------------------------
    def _default_adjustments(self, report: StepReport, evaluation: ReflectionResult) -> Dict[str, Any]:
        adjustments: Dict[str, Any] = {
            "severity": round(max(0.1, 1.0 - evaluation.confidence), 3),
            "retry_delay": report.retries + 1,
        }
        if report.error:
            adjustments["diagnostics"] = report.error
        if report.metrics:
            focus_metric = max(report.metrics.items(), key=lambda item: item[1])[0]
            adjustments["focus_metric"] = focus_metric
        return adjustments

    def _persist(self, record: Dict[str, Any]) -> None:
        if not self.memory:
            return
        try:
            payload = json.dumps(record, ensure_ascii=True)
            self.memory.store(payload, metadata={"category": "self_monitoring"})
        except Exception:  # pragma: no cover - defensive persistence
            pass
