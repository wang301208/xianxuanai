"""Metacognition utilities for monitoring reasoning traces.

This module implements a lightweight *monitor → control* loop that can watch
step-by-step reasoning traces (thoughts/plan/action) and emit corrective
suggestions and control flags.

Signals this monitor can use (best-effort):
- `confidence`: model self-confidence for the current step (0..1)
- `metadata.knowledge_hit` or related fields: whether retrieval found support
- `metadata.fatigue` (0..1): proxy for cognitive load / tiredness

Detected patterns (heuristics):
- repeated failed actions (stuck loops)
- no-progress cycles (same observation signature across steps)
- contradictory plans ("do X" + "do not X")
- very low confidence / sustained low confidence
- repeated knowledge misses (knowledge gap)
- sustained high fatigue (recommend rest / lower priority)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
import re
import time
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        number = float(value)
    except Exception:
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return float(number)


def _safe_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(int(value) != 0)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on", "y"}:
            return True
        if token in {"0", "false", "no", "off", "n"}:
            return False
    return None


def _compact_text(value: Any, *, max_chars: int) -> str:
    text = str(value or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _stable_fingerprint(value: Any, *, max_chars: int = 4_000) -> str:
    """Build a stable short fingerprint for `value` (for progress/loop checks)."""

    try:
        blob = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        blob = repr(value)
    blob = _compact_text(blob, max_chars=max_chars)
    digest = hashlib.sha1(blob.encode("utf-8", errors="ignore")).hexdigest()
    return digest


def _extract_action_name(action: Any) -> str:
    if action is None:
        return ""
    if isinstance(action, str):
        return action.strip()
    if isinstance(action, Mapping):
        direct = action.get("name") or action.get("action") or action.get("command")
        if isinstance(direct, str):
            return direct.strip()
        if isinstance(direct, Mapping):
            nested = direct.get("name") or direct.get("action")
            if isinstance(nested, str):
                return nested.strip()
        return str(direct or "").strip()
    return str(action).strip()


def _extract_knowledge_hit(metadata: Mapping[str, Any], observation: Any) -> Optional[bool]:
    """Return whether the current step has supporting knowledge (best-effort)."""

    for key in (
        "knowledge_hit",
        "knowledge_found",
        "knowledge_supported",
        "retrieval_hit",
        "memory_hit",
    ):
        flag = _safe_bool(metadata.get(key))
        if flag is not None:
            return flag

    for key in ("knowledge_hits", "memory_hits", "retrieval_hits", "hits"):
        hits = metadata.get(key)
        if hits is None:
            continue
        if isinstance(hits, (int, float)):
            return float(hits) > 0.0
        if isinstance(hits, (list, tuple, set)):
            return len(hits) > 0

    # Fallback: check observation for common shapes.
    if isinstance(observation, Mapping):
        for key in ("knowledge_hit", "retrieval_hit", "memory_hit"):
            flag = _safe_bool(observation.get(key))
            if flag is not None:
                return flag
        for key in ("knowledge_hits", "retrieval_hits", "memory_hits"):
            hits = observation.get(key)
            if isinstance(hits, (int, float)):
                return float(hits) > 0.0
            if isinstance(hits, (list, tuple, set)):
                return len(hits) > 0

    return None


def _extract_fatigue(metadata: Mapping[str, Any], observation: Any) -> Optional[float]:
    for key in ("fatigue", "sleepiness", "tiredness"):
        value = _safe_float(metadata.get(key), None)
        if value is not None:
            return float(max(0.0, min(1.0, value)))
    if isinstance(observation, Mapping):
        for key in ("fatigue", "sleepiness", "tiredness"):
            value = _safe_float(observation.get(key), None)
            if value is not None:
                return float(max(0.0, min(1.0, value)))
    return None


def _extract_query_hint(record: Mapping[str, Any]) -> str:
    meta = record.get("metadata")
    if isinstance(meta, Mapping):
        for key in ("query", "question", "prompt", "goal", "task"):
            value = meta.get(key)
            if value is None:
                continue
            token = str(value).strip()
            if token:
                return token
    thought = str(record.get("thought") or "").strip()
    if thought:
        return thought
    return ""


_NEGATION_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\bdo\s+not\b", re.IGNORECASE),
    re.compile(r"\bdon't\b", re.IGNORECASE),
    re.compile(r"\bnot\b", re.IGNORECASE),
    re.compile(r"\bnever\b", re.IGNORECASE),
    re.compile(r"\bavoid\b", re.IGNORECASE),
    re.compile(r"\u4e0d\u8981"),  # 不要
    re.compile(r"\u522b"),  # 别
    re.compile(r"\u907f\u514d"),  # 避免
    re.compile(r"\u7981\u6b62"),  # 禁止
)


def _plan_items(plan: Any) -> List[str]:
    if plan is None:
        return []
    if isinstance(plan, str):
        raw_lines = [line.strip() for line in plan.splitlines()]
        lines = []
        for line in raw_lines:
            if not line:
                continue
            line = re.sub(r"^[\-\*\d\.\)\]]+\s*", "", line).strip()
            if line:
                lines.append(line)
        return lines
    if isinstance(plan, (list, tuple, set)):
        lines: List[str] = []
        for item in plan:
            token = str(item or "").strip()
            if token:
                lines.append(token)
        return lines
    token = str(plan).strip()
    return [token] if token else []


def _canonicalise_plan_item(item: str) -> Tuple[str, bool]:
    text = str(item or "").strip()
    if not text:
        return "", False
    negated = any(pattern.search(text) for pattern in _NEGATION_PATTERNS)
    lowered = text.lower()
    lowered = re.sub(r"[^\w\s\u4e00-\u9fff]+", " ", lowered)
    for pattern in _NEGATION_PATTERNS:
        lowered = pattern.sub(" ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered, negated


@dataclass(frozen=True)
class MetaIssue:
    """A detected flaw/pattern in the reasoning trace."""

    kind: str
    severity: float
    message: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": float(self.severity),
            "message": self.message,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class MetaSuggestion:
    """A recommended meta-level corrective action."""

    kind: str
    reason: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "reason": self.reason, "payload": dict(self.payload)}


@dataclass
class MetaCognitionConfig:
    """Configuration for step-level metacognitive monitoring."""

    history_size: int = 64

    repeat_action_threshold: int = 3
    repeat_failure_threshold: int = 2
    no_progress_window: int = 4

    very_low_confidence_threshold: float = 0.15
    low_confidence_threshold: float = 0.35
    low_confidence_window: int = 3

    knowledge_miss_window: int = 4
    knowledge_miss_threshold: int = 2

    fatigue_window: int = 3
    fatigue_threshold: float = 0.8

    max_text_chars: int = 500


class MetaCognition:
    """Second-order monitor for reasoning traces and action loops."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = MetaCognitionConfig(**(config or {}))
        self._history: Deque[Dict[str, Any]] = deque(maxlen=max(1, int(self.config.history_size)))
        self._last_assessment: Dict[str, Any] = {}

    def reset(self) -> None:
        self._history.clear()
        self._last_assessment = {}

    def last_assessment(self) -> Dict[str, Any]:
        return dict(self._last_assessment)

    def observe_step(
        self,
        *,
        thoughts: Mapping[str, Any] | str | None = None,
        plan: Any = None,
        action: Any = None,
        observation: Any = None,
        success: bool | None = None,
        confidence: float | None = None,
        error: str | None = None,
        timestamp: float | None = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingest one reasoning/action step and return an assessment snapshot."""

        now = time.time() if timestamp is None else float(timestamp)
        action_name = _extract_action_name(action)

        thought_text = None
        thought_plan = None
        if isinstance(thoughts, Mapping):
            thought_text = thoughts.get("text") or thoughts.get("summary") or thoughts.get("reasoning")
            thought_plan = thoughts.get("plan")
        elif isinstance(thoughts, str):
            thought_text = thoughts

        plan_items = _plan_items(plan if plan is not None else thought_plan)
        obs_fp = _stable_fingerprint(observation)

        meta = dict(metadata or {})
        knowledge_hit = _extract_knowledge_hit(meta, observation)
        fatigue = _extract_fatigue(meta, observation)

        record = {
            "time": now,
            "action": action_name,
            "success": bool(success) if success is not None else None,
            "confidence": _safe_float(confidence, None),
            "error": str(error or "").strip() if error else "",
            "observation_fp": obs_fp,
            "thought": _compact_text(thought_text, max_chars=int(self.config.max_text_chars)) if thought_text else "",
            "plan": [str(item) for item in plan_items],
            "metadata": meta,
            "knowledge_hit": knowledge_hit,
            "fatigue": fatigue,
        }
        self._history.append(record)

        issues: List[MetaIssue] = []
        issues.extend(self._check_high_uncertainty_single())
        issues.extend(self._check_knowledge_gap())
        issues.extend(self._check_fatigue_high())
        issues.extend(self._check_repeated_failures())
        issues.extend(self._check_no_progress())
        issues.extend(self._check_plan_contradictions(plan_items))
        issues.extend(self._check_low_confidence_pattern())

        suggestions: List[MetaSuggestion] = []
        for issue in issues:
            suggestions.extend(self._suggest_from_issue(issue, record=record))

        control = self._build_control_flags(issues, suggestions)
        self_warning = control.get("self_warning")

        assessment = {
            "time": now,
            "issues": [issue.to_dict() for issue in issues],
            "suggestions": [s.to_dict() for s in suggestions],
            "self_warning": self_warning,
            "control": control,
            "latest": {
                "action": action_name,
                "success": record["success"],
                "confidence": record["confidence"],
                "knowledge_hit": knowledge_hit,
                "fatigue": fatigue,
            },
        }
        self._last_assessment = assessment
        return assessment

    # ------------------------------------------------------------------ #
    def _is_failure(self, entry: Mapping[str, Any]) -> bool:
        success = entry.get("success")
        if success is False:
            return True
        error_value = str(entry.get("error") or "").strip()
        if error_value:
            return True
        return False

    def _check_repeated_failures(self) -> List[MetaIssue]:
        threshold = max(2, int(self.config.repeat_action_threshold))
        min_failures = max(1, int(self.config.repeat_failure_threshold))
        recent = list(self._history)
        if not recent:
            return []

        last_action = str(recent[-1].get("action") or "").strip()
        if not last_action:
            return []

        consecutive: List[Dict[str, Any]] = []
        for entry in reversed(recent):
            if str(entry.get("action") or "").strip() != last_action:
                break
            consecutive.append(dict(entry))

        if len(consecutive) < threshold:
            return []

        failures = sum(1 for entry in consecutive if self._is_failure(entry))
        if failures < min_failures:
            return []

        severity = min(1.0, 0.4 + 0.15 * failures)
        return [
            MetaIssue(
                kind="repeated_action_failure",
                severity=severity,
                message=f"Action '{last_action}' failed repeatedly ({failures}/{len(consecutive)}).",
                evidence={"action": last_action, "consecutive": len(consecutive), "failures": failures},
            )
        ]

    def _check_no_progress(self) -> List[MetaIssue]:
        window = max(2, int(self.config.no_progress_window))
        recent = list(self._history)[-window:]
        if len(recent) < window:
            return []

        fp = recent[-1].get("observation_fp")
        if not fp:
            return []
        if any(entry.get("observation_fp") != fp for entry in recent):
            return []

        actions = [str(entry.get("action") or "") for entry in recent if entry.get("action")]
        return [
            MetaIssue(
                kind="no_progress_loop",
                severity=0.55,
                message="No progress detected (observation unchanged across recent steps).",
                evidence={"window": window, "actions": actions[-6:]},
            )
        ]

    def _check_plan_contradictions(self, plan_items: Sequence[str]) -> List[MetaIssue]:
        if not plan_items:
            return []

        seen: Dict[str, set[bool]] = {}
        conflicts: List[Tuple[str, List[str]]] = []
        for item in plan_items:
            canonical, negated = _canonicalise_plan_item(item)
            if not canonical or len(canonical) < 3:
                continue
            bucket = seen.setdefault(canonical, set())
            bucket.add(bool(negated))
            if bucket == {True, False}:
                evidence_items = [
                    p for p in plan_items if canonical and canonical in _canonicalise_plan_item(p)[0]
                ]
                conflicts.append((canonical, evidence_items[:6]))

        if not conflicts:
            return []

        return [
            MetaIssue(
                kind="plan_contradiction",
                severity=0.7,
                message="Contradictory plan items detected.",
                evidence={"conflicts": [{"canonical": c, "items": items} for c, items in conflicts]},
            )
        ]

    def _check_high_uncertainty_single(self) -> List[MetaIssue]:
        recent = list(self._history)
        if not recent:
            return []
        conf = recent[-1].get("confidence")
        if conf is None:
            return []
        try:
            value = float(conf)
        except Exception:
            return []
        threshold = float(self.config.very_low_confidence_threshold)
        if value >= threshold:
            return []
        severity = min(1.0, 0.6 + (threshold - value) * 0.8)
        return [
            MetaIssue(
                kind="high_uncertainty",
                severity=severity,
                message="Very low confidence detected; warn before taking irreversible actions.",
                evidence={"confidence": float(value), "threshold": threshold},
            )
        ]

    def _check_low_confidence_pattern(self) -> List[MetaIssue]:
        threshold = float(self.config.low_confidence_threshold)
        window = max(1, int(self.config.low_confidence_window))
        recent = [entry for entry in list(self._history)[-window:] if entry.get("confidence") is not None]
        if len(recent) < window:
            return []
        values = [float(entry["confidence"]) for entry in recent if entry.get("confidence") is not None]
        if not values:
            return []
        avg = sum(values) / float(len(values))
        if avg >= threshold:
            return []
        return [
            MetaIssue(
                kind="low_confidence_pattern",
                severity=0.5,
                message=f"Low confidence sustained over last {window} steps (avg={avg:.3f}).",
                evidence={"avg_confidence": float(avg), "window": window, "threshold": threshold},
            )
        ]

    def _check_knowledge_gap(self) -> List[MetaIssue]:
        window = max(1, int(self.config.knowledge_miss_window))
        threshold = max(1, int(self.config.knowledge_miss_threshold))
        recent = list(self._history)[-window:]
        if not recent:
            return []

        signals: List[bool] = []
        for entry in recent:
            hit = entry.get("knowledge_hit")
            if hit is None:
                continue
            signals.append(not bool(hit))  # miss=True

        if not signals:
            return []

        misses = sum(1 for miss in signals if miss)
        if misses < threshold:
            return []

        return [
            MetaIssue(
                kind="knowledge_gap",
                severity=0.6,
                message="Knowledge base has insufficient support for the current step.",
                evidence={"misses": misses, "window": window, "threshold": threshold},
            )
        ]

    def _check_fatigue_high(self) -> List[MetaIssue]:
        window = max(1, int(self.config.fatigue_window))
        threshold = float(self.config.fatigue_threshold)
        recent = [entry for entry in list(self._history)[-window:] if entry.get("fatigue") is not None]
        if len(recent) < window:
            return []
        values = [float(entry["fatigue"]) for entry in recent if entry.get("fatigue") is not None]
        if not values:
            return []
        avg = sum(values) / float(len(values))
        if avg < threshold:
            return []
        severity = min(1.0, 0.5 + (avg - threshold) * 0.8)
        return [
            MetaIssue(
                kind="fatigue_high",
                severity=severity,
                message=f"Fatigue is high (avg={avg:.3f}); consider resting or lowering task priority.",
                evidence={"avg_fatigue": float(avg), "window": window, "threshold": threshold},
            )
        ]

    # ------------------------------------------------------------------ #
    def _suggest_from_issue(self, issue: MetaIssue, *, record: Mapping[str, Any]) -> List[MetaSuggestion]:
        kind = issue.kind
        query_hint = _extract_query_hint(record)

        if kind in {"high_uncertainty", "low_confidence_pattern"}:
            return [
                MetaSuggestion(
                    kind="agent.self_warning",
                    reason=issue.message,
                    payload={
                        "message": "我不确定这样做是否正确",
                        "confidence": record.get("confidence"),
                    },
                ),
                MetaSuggestion(
                    kind="information.search",
                    reason="High uncertainty: look up supporting evidence before acting.",
                    payload={"query": query_hint} if query_hint else {},
                ),
                MetaSuggestion(
                    kind="request_human_feedback",
                    reason="High uncertainty: ask for confirmation when stakes are high.",
                    payload={},
                ),
            ]

        if kind == "knowledge_gap":
            return [
                MetaSuggestion(
                    kind="agent.self_warning",
                    reason=issue.message,
                    payload={
                        "message": "我目前缺乏这方面知识",
                        "knowledge_hit": record.get("knowledge_hit"),
                    },
                ),
                MetaSuggestion(
                    kind="knowledge.update",
                    reason=issue.message,
                    payload={"query": query_hint} if query_hint else {},
                ),
                MetaSuggestion(
                    kind="information.search",
                    reason="Knowledge gap: search docs/web or other sources.",
                    payload={"query": query_hint} if query_hint else {},
                ),
                MetaSuggestion(
                    kind="request_human_feedback",
                    reason="Knowledge gap: request guidance to avoid hallucination.",
                    payload={},
                ),
            ]

        if kind in {"repeated_action_failure", "no_progress_loop"}:
            avoid_action = issue.evidence.get("action") if isinstance(issue.evidence, Mapping) else None
            payload: Dict[str, Any] = {}
            if avoid_action:
                payload["avoid_action"] = avoid_action
            return [
                MetaSuggestion(
                    kind="planner.replan",
                    reason=issue.message,
                    payload={"hint": "Try a different approach or gather missing context."},
                ),
                MetaSuggestion(
                    kind="strategy.switch",
                    reason="Detected loop/failure pattern; switch to an alternative strategy.",
                    payload=payload,
                ),
            ]

        if kind == "plan_contradiction":
            return [
                MetaSuggestion(
                    kind="planner.resolve_contradiction",
                    reason=issue.message,
                    payload={"conflicts": issue.evidence.get("conflicts", [])},
                )
            ]

        if kind == "fatigue_high":
            return [
                MetaSuggestion(
                    kind="agent.rest",
                    reason=issue.message,
                    payload={"seconds": 60},
                ),
                MetaSuggestion(
                    kind="task.deprioritize",
                    reason="High fatigue can degrade performance; reduce urgency/priority temporarily.",
                    payload={},
                ),
            ]

        return []

    def _build_control_flags(
        self, issues: Sequence[MetaIssue], suggestions: Sequence[MetaSuggestion]
    ) -> Dict[str, Any]:
        issue_kinds = {issue.kind for issue in issues}
        suggestion_kinds = {s.kind for s in suggestions}

        should_warn = bool(issue_kinds & {"high_uncertainty", "low_confidence_pattern", "knowledge_gap"})
        should_replan = bool(issue_kinds & {"repeated_action_failure", "no_progress_loop", "plan_contradiction"})
        should_rest = "fatigue_high" in issue_kinds
        should_search = bool(suggestion_kinds & {"information.search", "knowledge.update"})

        warning = None
        if "knowledge_gap" in issue_kinds:
            warning = "我目前缺乏这方面知识"
        elif should_warn:
            warning = "我不确定这样做是否正确"

        return {
            "should_warn": should_warn,
            "should_replan": should_replan,
            "should_rest": should_rest,
            "should_search": should_search,
            "self_warning": warning,
        }


class MetaCognitiveMonitor:
    """Backwards-compatible monitor (legacy API).

    The original prototype updated calibration parameters based on
    `decision.confidence` vs `outcome.accuracy`. This wrapper keeps that surface
    available while delegating step-level checks to :class:`MetaCognition`.
    """

    def __init__(self) -> None:
        self.meta = MetaCognition()
        self.confidence_calibration = 0.5
        self.calibration_learning_rate = 0.1
        self.knowledge_gaps: set[str] = set()
        self.monitoring_accuracy = 0.0
        self.control_efficiency = 0.0

    def observe_step(self, **kwargs: Any) -> Dict[str, Any]:
        return self.meta.observe_step(**kwargs)

    def evaluate_decision(self, decision: Any, outcome: Any) -> None:
        confidence = _safe_float(getattr(decision, "confidence", None), 0.5) or 0.5
        accuracy = _safe_float(getattr(outcome, "accuracy", None), 0.5) or 0.5
        calibration_error = abs(float(confidence) - float(accuracy))
        adaptive_lr = float(self.calibration_learning_rate) * (1.0 + calibration_error)
        self.confidence_calibration = float(
            self.confidence_calibration * (1.0 - adaptive_lr) + (1.0 - calibration_error) * adaptive_lr
        )
        domain = getattr(decision, "domain", None)
        if accuracy < 0.5 and domain:
            self.knowledge_gaps.add(str(domain))
        self.monitoring_accuracy = 0.9 * float(self.monitoring_accuracy) + 0.1 * (1.0 - calibration_error)

    def get_control_suggestion(self) -> str:
        if float(self.confidence_calibration) < 0.3:
            return "need_more_verification"
        if len(self.knowledge_gaps) > 3:
            return "knowledge_gap_detected"
        return "ok"

    def reset(self) -> None:
        self.meta.reset()
        self.knowledge_gaps = set()


__all__ = [
    "MetaCognition",
    "MetaCognitionConfig",
    "MetaIssue",
    "MetaSuggestion",
    "MetaCognitiveMonitor",
]
