"""Self-awareness framework (prototype).

This module keeps a simple body representation and autobiographical memory. It
optionally attaches a step-level metacognition component that can monitor
reasoning traces (thoughts/plan/action) and emit corrective suggestions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .metacognition import MetaCognition

LLMCallable = Callable[[str], str]


def _tokenize(text: str) -> set[str]:
    blob = str(text or "").lower()
    ascii_tokens = re.findall(r"[a-z0-9_]+", blob)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", blob)
    return {t for t in ascii_tokens + cjk_chars if t}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


def _try_parse_json_object(text: str) -> Dict[str, Any] | None:
    blob = str(text or "").strip()
    if not blob:
        return None
    start = blob.find("{")
    end = blob.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(blob[start : end + 1])
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        number = float(default)
    if number != number or number in (float("inf"), float("-inf")):
        number = float(default)
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return float(number)


def _infer_success(outcome: str) -> bool | None:
    text = str(outcome or "").strip().lower()
    if not text:
        return None
    if any(token in text for token in ("fail", "failed", "error", "exception", "失败", "错误", "异常")):
        return False
    if any(token in text for token in ("success", "succeeded", "completed", "done", "成功", "完成")):
        return True
    return None


@dataclass
class BodyRepresentation:
    """Body-state representation used for basic self tracking."""

    limb_positions: Dict[str, float]
    sensory_inputs: Dict[str, float]

    def update_from_sensors(self, sensor_data: Mapping[str, Any]) -> None:
        """Update the body model with a new sensor snapshot."""

        for key, value in sensor_data.items():
            if key not in self.limb_positions:
                continue
            try:
                number = float(value)
            except Exception:
                continue
            self.limb_positions[key] = 0.9 * self.limb_positions[key] + 0.1 * number


@dataclass
class AutobiographicalMemory:
    """Minimal autobiographical memory store."""

    episodes: List[Dict[str, Any]]
    self_concept: Dict[str, float]

    def retrieve_episode(self, query: Mapping[str, Any]) -> Dict[str, Any]:
        """Retrieve the most similar episode (best-effort)."""

        if not self.episodes:
            return {}
        similarities = [(ep, self._calculate_similarity(ep, query)) for ep in self.episodes]
        return max(similarities, key=lambda pair: pair[1])[0]

    def add_reflection(
        self,
        *,
        task: str,
        outcome: str,
        reflection: str,
        lessons: Sequence[str] | None = None,
        checklist: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
        timestamp: float | None = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Append a self-reflection note into autobiographical memory."""

        now = time.time() if timestamp is None else float(timestamp)
        record: Dict[str, Any] = {
            "type": "self_reflection",
            "time": now,
            "task": str(task or "").strip(),
            "outcome": str(outcome or "").strip(),
            "reflection": str(reflection or "").strip(),
            "lessons": [str(x).strip() for x in (lessons or []) if str(x).strip()],
            "checklist": [str(x).strip() for x in (checklist or []) if str(x).strip()],
            "tags": [str(x).strip() for x in (tags or []) if str(x).strip()],
            "self_concept": dict(self.self_concept),
            "metadata": dict(metadata or {}),
        }
        self.episodes.append(record)
        return record

    def retrieve_reflection_notes(
        self,
        query_text: str,
        *,
        top_k: int = 3,
        min_similarity: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Retrieve past self-reflection notes relevant to *query_text*."""

        query = str(query_text or "").strip()
        if not query:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored: List[tuple[float, Dict[str, Any]]] = []
        for ep in self.episodes:
            if not isinstance(ep, dict) or ep.get("type") != "self_reflection":
                continue
            haystack = " ".join(
                str(part or "")
                for part in (
                    ep.get("task"),
                    ep.get("outcome"),
                    ep.get("reflection"),
                    " ".join(ep.get("tags", []) if isinstance(ep.get("tags"), list) else []),
                )
            )
            score = _jaccard(query_tokens, _tokenize(haystack))
            if score >= float(min_similarity):
                scored.append((score, ep))

        if not scored:
            return []
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [dict(ep, similarity=score) for score, ep in scored[: max(1, int(top_k))]]

    @staticmethod
    def _calculate_similarity(episode: Mapping[str, Any], query: Mapping[str, Any]) -> float:
        score = 0.0
        for key, value in query.items():
            if key in episode:
                score += 0.5 if episode[key] == value else 0.1
        return score / len(query) if query else 0.0


class SelfAwarenessFramework:
    """Combine self representation, autobiographical memory, and metacognition."""

    def __init__(
        self,
        *,
        enable_metacognition: bool = True,
        metacognition: MetaCognition | None = None,
    ) -> None:
        self.body_model = BodyRepresentation(
            limb_positions={"arm": 0.5, "head": 0.0},
            sensory_inputs={},
        )
        self.memory = AutobiographicalMemory(
            episodes=[],
            self_concept={
                "agency": 0.7,
                "identity": 0.8,
                "competence_confidence": 0.6,
                "curiosity": 0.5,
                "frustration": 0.2,
                "achievement": 0.2,
            },
        )
        self.meta_cognition: MetaCognition | None = metacognition if enable_metacognition else None
        if self.meta_cognition is None and enable_metacognition:
            self.meta_cognition = MetaCognition()
        self.metacognition_state: Dict[str, Any] = {}

        self._recent_outcomes: Deque[float] = deque(maxlen=30)
        self._recent_step_confidence: Deque[float] = deque(maxlen=30)
        self._success_streak = 0
        self._failure_streak = 0
        self._last_curiosity_drive: float | None = None

    def update_self_state(self, sensor_data: Mapping[str, Any] | None = None) -> None:
        """Update self state and append a new autobiographical snapshot."""

        if sensor_data:
            self.body_model.update_from_sensors(sensor_data)

        now = time.time()
        current_self: Dict[str, Any] = {
            "body": dict(self.body_model.limb_positions),
            "time": self._get_internal_clock(),
            "timestamp": now,
            "self_concept": dict(self.memory.self_concept),
        }
        if self.metacognition_state:
            current_self["metacognition"] = dict(self.metacognition_state)

        self.memory.episodes.append(current_self)
        self._update_self_concept()

    def observe_reasoning_step(
        self,
        *,
        thoughts: Mapping[str, Any] | str | None = None,
        plan: Any = None,
        action: Any = None,
        observation: Any = None,
        success: bool | None = None,
        confidence: float | None = None,
        error: str | None = None,
        metadata: Optional[Mapping[str, Any]] = None,
        timestamp: float | None = None,
    ) -> Dict[str, Any]:
        """Observe one decision/reasoning step and run metacognitive checks."""

        if self.meta_cognition is None:
            return {}

        now = time.time() if timestamp is None else float(timestamp)
        assessment = self.meta_cognition.observe_step(
            thoughts=thoughts,
            plan=plan,
            action=action,
            observation=observation,
            success=success,
            confidence=confidence,
            error=error,
            timestamp=now,
            metadata=metadata,
        )
        self.metacognition_state = dict(assessment or {})
        self._update_self_concept_from_step(success=success, confidence=confidence, metadata=metadata)

        # Store as a dedicated memory entry for later retrieval.
        self.memory.episodes.append(
            {
                "type": "metacognition",
                "time": now,
                "assessment": dict(self.metacognition_state),
                "self_concept": dict(self.memory.self_concept),
            }
        )
        return dict(self.metacognition_state)

    def observe_task_outcome(
        self,
        *,
        task: str,
        success: bool,
        confidence: float | None = None,
        curiosity_drive: float | None = None,
        metadata: Optional[Mapping[str, Any]] = None,
        timestamp: float | None = None,
    ) -> Dict[str, Any]:
        """Record a task-level outcome and update the self-concept accordingly."""

        now = time.time() if timestamp is None else float(timestamp)
        meta: Dict[str, Any] = dict(metadata or {})
        if curiosity_drive is not None:
            meta.setdefault("curiosity_drive", curiosity_drive)
        self._update_self_concept_from_step(success=bool(success), confidence=confidence, metadata=meta)

        record = {
            "type": "task_outcome",
            "time": now,
            "task": str(task or "").strip(),
            "success": bool(success),
            "confidence": _clamp01(confidence, default=0.0) if confidence is not None else None,
            "curiosity_drive": curiosity_drive,
            "self_concept": dict(self.memory.self_concept),
            "metadata": meta,
        }
        self.memory.episodes.append(record)
        return dict(record)

    def reflect_after_task(
        self,
        *,
        task: str,
        outcome: str,
        trace: Optional[Sequence[Mapping[str, Any]]] = None,
        llm: LLMCallable | None = None,
        timestamp: float | None = None,
        tags: Optional[Sequence[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a post-task reflection and store it into autobiographical memory.

        The reflection is LLM-backed when *llm* is provided. Otherwise a
        deterministic fallback is used.
        """

        now = time.time() if timestamp is None else float(timestamp)
        trace = list(trace or [])[-10:]

        prompt = (
            "请回顾刚才的任务执行过程，并给出可复用的改进建议。\n"
            "输出严格 JSON：\n"
            "{\n"
            '  "reflection": "一句话总评",\n'
            '  "lessons": ["经验教训1"],\n'
            '  "checklist": ["下次执行前检查A"],\n'
            '  "tags": ["domain_or_error_type"]\n'
            "}\n\n"
            "输入：\n"
            + json.dumps(
                {
                    "task": task,
                    "outcome": outcome,
                    "recent_trace": trace,
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
        )

        reflection_text = ""
        lessons: List[str] = []
        checklist: List[str] = []
        parsed_tags: List[str] = []

        if llm is not None:
            try:
                raw = (llm(prompt) or "").strip()
                data = _try_parse_json_object(raw)
                if isinstance(data, dict):
                    reflection_text = str(data.get("reflection") or "").strip() or str(raw).strip()
                    lessons = [str(x).strip() for x in (data.get("lessons") or []) if str(x).strip()]
                    checklist = [str(x).strip() for x in (data.get("checklist") or []) if str(x).strip()]
                    parsed_tags = [str(x).strip() for x in (data.get("tags") or []) if str(x).strip()]
                else:
                    reflection_text = str(raw).strip()
            except Exception:
                reflection_text = ""

        if not reflection_text:
            reflection_text = "本次任务结束，建议回顾目标、检查关键假设，并记录可复用的检查清单。"
        if not lessons:
            lessons = [
                "先明确成功标准与可验证检查点。",
                "不确定时先检索已有知识与历史经验。",
            ]
        if not checklist:
            checklist = [
                "复述目标与约束，确认理解无误。",
                "对关键输入/输出做有效性检查（空值、格式、范围）。",
            ]

        merged_tags = [str(x).strip() for x in (tags or parsed_tags) if str(x).strip()]
        inferred_success = _infer_success(outcome)
        if inferred_success is not None:
            self._update_self_concept_from_step(success=inferred_success, confidence=None, metadata=None)
        record = self.memory.add_reflection(
            task=task,
            outcome=outcome,
            reflection=reflection_text,
            lessons=lessons,
            checklist=checklist,
            tags=merged_tags,
            timestamp=now,
            metadata=metadata,
        )
        return dict(record)

    def retrieve_reflection_hints(
        self,
        query_text: str,
        *,
        top_k: int = 3,
        min_similarity: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant self-reflection notes for a new task/query."""

        return self.memory.retrieve_reflection_notes(
            query_text,
            top_k=top_k,
            min_similarity=min_similarity,
        )

    def _get_internal_clock(self) -> float:
        """Internal time perception stub."""

        return float(np.random.uniform(0.0, 1.0))

    def _update_self_concept(self) -> None:
        """Update self concept based on recent experiences."""

        recent_eps = [ep for ep in self.memory.episodes[-10:] if isinstance(ep, Mapping)]
        denom = float(len(recent_eps) or 1)
        recent_agency = sum(float(ep.get("action_intensity", 0.5) or 0.5) for ep in recent_eps) / denom
        self.memory.self_concept["agency"] = _clamp01(
            0.9 * float(self.memory.self_concept.get("agency", 0.7)) + 0.1 * float(recent_agency),
            default=0.7,
        )

        success_rate: float | None = None
        if self._recent_outcomes:
            success_rate = sum(self._recent_outcomes) / float(len(self._recent_outcomes))
            success_rate = _clamp01(success_rate, default=0.5)

        avg_step_conf: float | None = None
        if self._recent_step_confidence:
            avg_step_conf = sum(self._recent_step_confidence) / float(len(self._recent_step_confidence))
            avg_step_conf = _clamp01(avg_step_conf, default=0.5)

        competence_old = float(self.memory.self_concept.get("competence_confidence", 0.6))
        competence_target = competence_old
        if success_rate is not None and avg_step_conf is not None:
            competence_target = 0.65 * success_rate + 0.35 * avg_step_conf
        elif success_rate is not None:
            competence_target = success_rate
        elif avg_step_conf is not None:
            competence_target = avg_step_conf
        self.memory.self_concept["competence_confidence"] = _clamp01(
            0.85 * competence_old + 0.15 * competence_target,
            default=0.6,
        )

        curiosity_old = float(self.memory.self_concept.get("curiosity", 0.5))
        if self._last_curiosity_drive is not None:
            drive = float(self._last_curiosity_drive)
            if -1.0 <= drive <= 1.0:
                curiosity_target = (max(-1.0, min(1.0, drive)) + 1.0) / 2.0
            else:
                curiosity_target = _clamp01(drive, default=0.5)
            self.memory.self_concept["curiosity"] = _clamp01(
                0.9 * curiosity_old + 0.1 * curiosity_target,
                default=0.5,
            )
        else:
            self.memory.self_concept["curiosity"] = _clamp01(0.98 * curiosity_old + 0.02 * 0.5, default=0.5)

        rate = float(success_rate) if success_rate is not None else 0.5
        streak_s = min(int(self._success_streak), 5)
        streak_f = min(int(self._failure_streak), 5)
        frustration_target = _clamp01(0.6 * (1.0 - rate) + 0.08 * float(streak_f), default=0.2)
        achievement_target = _clamp01(0.6 * rate + 0.08 * float(streak_s), default=0.2)

        frustration_old = float(self.memory.self_concept.get("frustration", 0.2))
        achievement_old = float(self.memory.self_concept.get("achievement", 0.2))
        self.memory.self_concept["frustration"] = _clamp01(
            0.85 * frustration_old + 0.15 * frustration_target,
            default=0.2,
        )
        self.memory.self_concept["achievement"] = _clamp01(
            0.85 * achievement_old + 0.15 * achievement_target,
            default=0.2,
        )

    def _update_self_concept_from_step(
        self,
        *,
        success: bool | None,
        confidence: float | None,
        metadata: Optional[Mapping[str, Any]],
    ) -> None:
        if success is not None:
            self._recent_outcomes.append(1.0 if bool(success) else 0.0)
            if bool(success):
                self._success_streak += 1
                self._failure_streak = 0
            else:
                self._failure_streak += 1
                self._success_streak = 0
        if confidence is not None:
            self._recent_step_confidence.append(_clamp01(confidence, default=0.0))
        if metadata is not None:
            drive = metadata.get("curiosity_drive")
            if isinstance(drive, (int, float)):
                self._last_curiosity_drive = float(drive)
            elif isinstance(drive, str):
                try:
                    self._last_curiosity_drive = float(drive)
                except Exception:
                    pass
        self._update_self_concept()

    def decision_bias(self) -> Dict[str, float]:
        """Return simple decision preferences derived from current self-concept."""

        concept = self.memory.self_concept
        competence = _clamp01(concept.get("competence_confidence", 0.6), default=0.6)
        curiosity = _clamp01(concept.get("curiosity", 0.5), default=0.5)
        frustration = _clamp01(concept.get("frustration", 0.2), default=0.2)
        achievement = _clamp01(concept.get("achievement", 0.2), default=0.2)

        # Higher competence/achievement and curiosity => more exploration.
        exploration_bias = _clamp01(0.6 * curiosity + 0.25 * competence + 0.15 * achievement - 0.35 * frustration)
        # Low competence or high frustration => more conservative.
        caution_level = _clamp01((1.0 - competence) * 0.6 + frustration * 0.5 - achievement * 0.2)
        return {
            "exploration_bias": exploration_bias,
            "caution_level": caution_level,
        }

    def daily_self_check(
        self,
        *,
        now: float | None = None,
        llm: LLMCallable | None = None,
        export_path: str | Path | None = None,
        include_raw: bool = False,
    ) -> Dict[str, Any]:
        """Review today's activity and store a diary-like self-check report.

        This is intentionally lightweight and deterministic by default. When *llm*
        is provided, the report can include an LLM-generated narrative summary.
        """

        ts = time.time() if now is None else float(now)
        local_dt = datetime.fromtimestamp(ts)
        day_start = local_dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        date_str = local_dt.date().isoformat()

        episodes_today: List[Dict[str, Any]] = []
        for ep in list(self.memory.episodes):
            if not isinstance(ep, Mapping):
                continue
            ep_ts = ep.get("timestamp")
            if isinstance(ep_ts, (int, float)) and float(ep_ts) > 1_000_000.0:
                timestamp_value = float(ep_ts)
            else:
                raw_time = ep.get("time")
                timestamp_value = float(raw_time) if isinstance(raw_time, (int, float)) and float(raw_time) > 1_000_000.0 else None  # type: ignore[assignment]
            if timestamp_value is None:
                continue
            if day_start <= timestamp_value <= ts:
                episodes_today.append(dict(ep))

        task_outcomes = [ep for ep in episodes_today if ep.get("type") == "task_outcome"]
        reflections = [ep for ep in episodes_today if ep.get("type") == "self_reflection"]
        metacognition = [ep for ep in episodes_today if ep.get("type") == "metacognition"]
        human_feedback = [ep for ep in episodes_today if ep.get("type") == "human_feedback"]

        success_count = sum(1 for ep in task_outcomes if ep.get("success") is True)
        failure_count = sum(1 for ep in task_outcomes if ep.get("success") is False)
        total_tasks = len(task_outcomes)
        success_rate = success_count / total_tasks if total_tasks else None

        issue_counts: Dict[str, int] = {}
        warning_counts: Dict[str, int] = {}
        for ep in metacognition:
            assessment = ep.get("assessment")
            if not isinstance(assessment, Mapping):
                continue
            for issue in assessment.get("issues", []) if isinstance(assessment.get("issues"), list) else []:
                if not isinstance(issue, Mapping):
                    continue
                kind = str(issue.get("kind") or "").strip() or "issue"
                issue_counts[kind] = issue_counts.get(kind, 0) + 1
            warning = assessment.get("self_warning")
            if isinstance(warning, str) and warning.strip():
                warning_counts[warning.strip()] = warning_counts.get(warning.strip(), 0) + 1

        tag_counts: Dict[str, int] = {}
        lesson_counts: Dict[str, int] = {}
        checklist_counts: Dict[str, int] = {}
        for ep in reflections:
            for tag in ep.get("tags", []) if isinstance(ep.get("tags"), list) else []:
                token = str(tag).strip()
                if token:
                    tag_counts[token] = tag_counts.get(token, 0) + 1
            for item in ep.get("lessons", []) if isinstance(ep.get("lessons"), list) else []:
                text = str(item).strip()
                if text:
                    lesson_counts[text] = lesson_counts.get(text, 0) + 1
            for item in ep.get("checklist", []) if isinstance(ep.get("checklist"), list) else []:
                text = str(item).strip()
                if text:
                    checklist_counts[text] = checklist_counts.get(text, 0) + 1

        def _top_k(counter: Mapping[str, int], k: int) -> List[Dict[str, Any]]:
            items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
            return [{"value": key, "count": count} for key, count in items[: max(0, int(k))]]

        # Determine start/end self-concept snapshots for the day (best-effort).
        concept_start: Dict[str, float] | None = None
        concept_end: Dict[str, float] | None = None
        for ep in episodes_today:
            concept = ep.get("self_concept")
            if isinstance(concept, Mapping) and concept_start is None:
                concept_start = {str(k): _clamp01(v, default=0.0) for k, v in concept.items()}
            if isinstance(concept, Mapping):
                concept_end = {str(k): _clamp01(v, default=0.0) for k, v in concept.items()}
        if concept_start is None:
            concept_start = {str(k): _clamp01(v, default=0.0) for k, v in self.memory.self_concept.items()}
        if concept_end is None:
            concept_end = {str(k): _clamp01(v, default=0.0) for k, v in self.memory.self_concept.items()}

        concept_delta: Dict[str, float] = {}
        for key in sorted(set(concept_start) | set(concept_end)):
            a = float(concept_start.get(key, 0.0) or 0.0)
            b = float(concept_end.get(key, 0.0) or 0.0)
            concept_delta[key] = round(b - a, 4)

        competence = _clamp01(concept_end.get("competence_confidence", 0.6), default=0.6)
        frustration = _clamp01(concept_end.get("frustration", 0.2), default=0.2)

        recommendations: List[str] = []
        if success_rate is not None:
            if success_rate < 0.6 and competence > 0.7:
                recommendations.append("可能存在过度自信：建议降低主观置信度，并增加外部验证/检索。")
            if success_rate > 0.8 and competence < 0.4:
                recommendations.append("可能存在低估能力：可适度提升自信并减少不必要的保守回退。")
            if success_rate < 0.5:
                recommendations.append("今日失败率偏高：优先复盘高频失败模式，形成可复用的检查清单。")
        if frustration > 0.7:
            recommendations.append("沮丧度偏高：建议降低任务并发/复杂度，必要时请求人类确认关键决策。")
        if issue_counts.get("knowledge_gap", 0) >= 2:
            recommendations.append("知识缺口信号频繁：建议补齐该领域知识并建立检索优先策略。")
        if not recommendations:
            recommendations.append("整体表现稳定：保持当前策略，并持续记录可复用的经验教训。")

        narrative = ""
        if llm is not None:
            try:
                prompt = (
                    "请基于以下 JSON 生成一段简短的“每日自省日记”，包含：\n"
                    "- 今日完成情况（做得好的/不好的）\n"
                    "- 主要失败模式与改进点\n"
                    "- 明日一条最重要的行动建议\n\n"
                    "输出纯文本，不要 JSON。\n\n"
                    + json.dumps(
                        {
                            "date": date_str,
                            "task_stats": {"total": total_tasks, "success": success_count, "failure": failure_count, "success_rate": success_rate},
                            "issue_counts": issue_counts,
                            "top_tags": _top_k(tag_counts, 5),
                            "self_concept_end": concept_end,
                            "recommendations": recommendations,
                            "human_feedback_count": len(human_feedback),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                )
                narrative = str(llm(prompt) or "").strip()
            except Exception:
                narrative = ""

        report: Dict[str, Any] = {
            "type": "daily_self_check",
            "date": date_str,
            "timestamp": ts,
            "window": {"start": day_start, "end": ts},
            "task_stats": {
                "total": total_tasks,
                "success": success_count,
                "failure": failure_count,
                "success_rate": round(success_rate, 4) if success_rate is not None else None,
            },
            "metacognition": {
                "issue_counts": dict(sorted(issue_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
                "warning_counts": dict(sorted(warning_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            },
            "reflections": {
                "top_tags": _top_k(tag_counts, 6),
                "top_lessons": _top_k(lesson_counts, 6),
                "top_checklist": _top_k(checklist_counts, 8),
            },
            "self_concept": {
                "start": concept_start,
                "end": concept_end,
                "delta": concept_delta,
            },
            "recommendations": recommendations,
            "human_feedback": human_feedback if include_raw else [{"time": ep.get("time"), "notes": ep.get("notes"), "rating": ep.get("rating")} for ep in human_feedback],  # noqa: E501
            "narrative": narrative or None,
        }

        self.memory.episodes.append(dict(report))

        if export_path is not None:
            try:
                path = Path(export_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

        return dict(report)

    def apply_human_feedback(
        self,
        *,
        rating: float | None = None,
        notes: str | None = None,
        calibration: Optional[Mapping[str, Any]] = None,
        deltas: Optional[Mapping[str, Any]] = None,
        learning_rate: float = 0.25,
        timestamp: float | None = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Apply human supervision feedback to calibrate self-concept.

        Parameters
        ----------
        rating:
            Human evaluation (0..1 or 1..5); stored for auditing.
        calibration:
            Optional absolute targets for self-concept keys, e.g.
            ``{"competence_confidence": 0.4}``.
        deltas:
            Optional relative adjustments, e.g. ``{"frustration": +0.1}``.
        learning_rate:
            How strongly to move towards *calibration* targets (0..1).
        """

        now = time.time() if timestamp is None else float(timestamp)
        before = dict(self.memory.self_concept)

        lr = _clamp01(learning_rate, default=0.25)
        if calibration:
            for key, value in calibration.items():
                if key not in before:
                    continue
                target = _clamp01(value, default=float(before.get(key, 0.0) or 0.0))
                updated = (1.0 - lr) * float(before.get(key, 0.0) or 0.0) + lr * float(target)
                self.memory.self_concept[key] = _clamp01(updated, default=float(before.get(key, 0.0) or 0.0))

        if deltas:
            for key, value in deltas.items():
                if key not in self.memory.self_concept:
                    continue
                try:
                    delta = float(value)
                except Exception:
                    continue
                updated = float(self.memory.self_concept.get(key, 0.0) or 0.0) + delta
                self.memory.self_concept[key] = _clamp01(updated, default=float(before.get(key, 0.0) or 0.0))

        after = dict(self.memory.self_concept)
        record = {
            "type": "human_feedback",
            "time": now,
            "rating": float(rating) if rating is not None else None,
            "notes": str(notes or "").strip() or None,
            "calibration": dict(calibration or {}),
            "deltas": dict(deltas or {}),
            "learning_rate": lr,
            "self_concept_before": before,
            "self_concept_after": after,
            "metadata": dict(metadata or {}),
        }
        self.memory.episodes.append(dict(record))
        return dict(record)

    def recognize_self(self, mirror_input: Mapping[str, Any]) -> bool:
        """Simple mirror self-recognition check (toy heuristic)."""

        items = list(mirror_input.items())
        if not items:
            return False
        diffs: List[float] = []
        for key, value in items:
            try:
                observed = float(value)
            except Exception:
                continue
            diffs.append(abs(observed - float(self.body_model.limb_positions.get(str(key), 0.0))))
        if not diffs:
            return False
        motion_similarity = sum(diffs) / float(len(diffs))
        return motion_similarity < 0.2


__all__ = ["SelfAwarenessFramework", "AutobiographicalMemory", "BodyRepresentation"]
