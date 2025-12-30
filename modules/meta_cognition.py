"""Meta-level controller for long-horizon introspection and strategy resets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional
from collections import defaultdict, deque
import time


@dataclass
class MetaSignal:
    """High-level action suggested by the meta controller."""

    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)


class MetaCognitionController:
    """Monitor long-term performance and suggest strategy/goal adjustments."""

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        low_confidence_threshold: float = 0.35,
        low_confidence_window: int = 5,
        knowledge_gap_threshold: int = 2,
        history_limit: int = 1000,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.low_confidence_threshold = float(low_confidence_threshold)
        self.low_confidence_window = max(1, int(low_confidence_window))
        self.knowledge_gap_threshold = max(1, int(knowledge_gap_threshold))
        self._task_failures: Dict[str, deque[bool]] = defaultdict(lambda: deque(maxlen=failure_threshold))
        self._task_confidence: Dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.low_confidence_window))
        self._task_knowledge_gaps: Dict[str, deque[bool]] = defaultdict(lambda: deque(maxlen=max(3, knowledge_gap_threshold)))
        self._meta_log: List[Dict[str, Any]] = []
        self._history_limit = history_limit

    # ------------------------------------------------------------------ #
    def record_task_outcome(self, task_id: str, success: bool, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Ingest a task outcome (success/failure) with optional context."""

        task_id = str(task_id or "task").strip() or "task"
        meta = dict(metadata or {})
        self._task_failures[task_id].append(bool(success))
        confidence = meta.get("confidence")
        try:
            if confidence is not None:
                self._task_confidence[task_id].append(float(confidence))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pass
        knowledge_gap = meta.get("knowledge_gap") or meta.get("knowledge_miss") or False
        self._task_knowledge_gaps[task_id].append(bool(knowledge_gap))
        entry = {
            "task_id": task_id,
            "success": bool(success),
            "metadata": meta,
            "timestamp": time.time(),
        }
        self._meta_log.append(entry)
        if len(self._meta_log) > self._history_limit:
            self._meta_log = self._meta_log[-self._history_limit :]

    # ------------------------------------------------------------------ #
    def record_cycle_review(
        self,
        *,
        task_id: str,
        decision: Mapping[str, Any] | None = None,
        outcome: Mapping[str, Any] | None = None,
        expected: Mapping[str, Any] | None = None,
    ) -> None:
        """Store a richer per-cycle introspection record (best-effort)."""

        payload = {
            "cycle_review": {
                "task_id": str(task_id or "task").strip() or "task",
                "decision": dict(decision or {}),
                "outcome": dict(outcome or {}),
                "expected": dict(expected or {}),
                "timestamp": time.time(),
            }
        }
        self._meta_log.append(payload)
        if len(self._meta_log) > self._history_limit:
            self._meta_log = self._meta_log[-self._history_limit :]

    # ------------------------------------------------------------------ #
    def record_human_feedback(
        self,
        *,
        task_id: str,
        prompt: str,
        agent_response: str,
        correct_response: str | None = None,
        rating: float | None = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Ingest supervised feedback to accelerate self-correction."""

        entry = {
            "human_feedback": {
                "task_id": str(task_id or "task").strip() or "task",
                "prompt": str(prompt or "").strip(),
                "agent_response": str(agent_response or "").strip(),
                "correct_response": str(correct_response or "").strip() if correct_response is not None else None,
                "rating": float(rating) if rating is not None else None,
                "metadata": dict(metadata or {}),
                "timestamp": time.time(),
            }
        }
        self._meta_log.append(entry)
        if len(self._meta_log) > self._history_limit:
            self._meta_log = self._meta_log[-self._history_limit :]

    # ------------------------------------------------------------------ #
    def record_regressions(self, regressions: Iterable[Dict[str, Any]]) -> None:
        """Store regression annotations coming from evolution/monitoring."""

        for regression in regressions:
            self._meta_log.append({"regression": dict(regression), "timestamp": time.time()})
        if len(self._meta_log) > self._history_limit:
            self._meta_log = self._meta_log[-self._history_limit :]

    # ------------------------------------------------------------------ #
    def analyse(self) -> List[MetaSignal]:
        """Return a list of high-level actions based on failures and regressions."""

        signals: List[MetaSignal] = []
        for task_id, outcomes in self._task_failures.items():
            if len(outcomes) < self.failure_threshold:
                continue
            if all(not ok for ok in outcomes):
                signals.append(
                    MetaSignal(
                        kind="replan_task",
                        payload={"task_id": task_id, "reason": "repeated_failures"},
                    )
                )
                signals.append(
                    MetaSignal(
                        kind="schedule_skill_learning",
                        payload={"task_id": task_id},
                    )
                )

        for task_id, samples in self._task_confidence.items():
            if len(samples) < self.low_confidence_window:
                continue
            avg = sum(samples) / max(1, len(samples))
            if avg < self.low_confidence_threshold:
                signals.append(
                    MetaSignal(
                        kind="request_human_feedback",
                        payload={
                            "task_id": task_id,
                            "reason": "low_confidence",
                            "avg_confidence": float(avg),
                        },
                    )
                )

        for task_id, gaps in self._task_knowledge_gaps.items():
            if len(gaps) < self.knowledge_gap_threshold:
                continue
            if sum(1 for g in gaps if g) >= self.knowledge_gap_threshold:
                signals.append(
                    MetaSignal(
                        kind="schedule_knowledge_learning",
                        payload={"task_id": task_id, "reason": "repeated_knowledge_gaps"},
                    )
                )

        recent_regressions = [
            entry.get("regression")
            for entry in self._meta_log[-20:]
            if isinstance(entry, dict) and "regression" in entry
        ]
        if recent_regressions:
            signals.append(
                MetaSignal(
                    kind="trigger_evolution",
                    payload={"reason": "regression_detected", "count": len(recent_regressions)},
                )
            )
        return signals

    # ------------------------------------------------------------------ #
    def meta_log(self) -> List[Dict[str, Any]]:
        """Return accumulated meta-cognition log."""

        return list(self._meta_log)


__all__ = ["MetaCognitionController", "MetaSignal"]
