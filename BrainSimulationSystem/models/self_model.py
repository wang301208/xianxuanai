"""
Self-awareness and self-model module providing metacognitive monitoring.

The module aggregates signals from perception, decision making, motivation and
meta-reasoning to maintain an internal belief state about the agent. It produces
succinct introspective reports that can be consumed by downstream planning or
communication layers.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class SelfModelConfig:
    """Configuration options for the self-awareness module."""

    window_size: int = 64
    report_history: int = 8
    introspection_interval: float = 1_000.0  # in simulation time units (e.g. ms)
    min_confidence: float = 0.35
    anomaly_threshold: float = 0.6
    enable_text_report: bool = True
    track_subsystems: Tuple[str, ...] = (
        "decision",
        "memory",
        "attention",
        "emotion",
    )


@dataclass
class SelfModelState:
    """Internal state for the self-awareness module."""

    last_report_time: float = 0.0
    cumulative_errors: int = 0
    total_observations: int = 0
    rolling_confidence: float = 0.5
    belief_state: Dict[str, Any] = field(default_factory=dict)


class SelfAwarenessModule:
    """Maintains a self-model and produces introspective reports."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = SelfModelConfig(**(config or {}))
        self.state = SelfModelState()
        self.observation_window: Deque[Dict[str, Any]] = deque(maxlen=int(self.config.window_size))
        self.reports: Deque[Dict[str, Any]] = deque(maxlen=int(self.config.report_history))

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.state = SelfModelState()
        self.observation_window.clear()
        self.reports.clear()

    def observe(
        self,
        *,
        time_point: float,
        goals: Iterable[Any],
        decision_result: Optional[Dict[str, Any]],
        attention_focus: Optional[Iterable[Dict[str, Any]]],
        attention_scores: Optional[Iterable[Dict[str, Any]]] = None,
        motivation: Optional[Dict[str, float]] = None,
        emotion_state: Optional[Dict[str, Any]] = None,
        memory_result: Optional[Dict[str, Any]] = None,
        meta_analysis: Optional[Dict[str, Any]] = None,
        reward: Optional[float] = None,
        plan_result: Optional[Dict[str, Any]] = None,
        working_memory: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update the self-model with the latest cognitive signals."""

        self.state.total_observations += 1
        decision_confidence = 0.0
        decision_choice = None
        decision_error_flag = False

        if isinstance(decision_result, dict):
            confidence_val = decision_result.get("confidence")
            if isinstance(confidence_val, (int, float)):
                decision_confidence = float(np.clip(confidence_val, 0.0, 1.0))
            decision_choice = decision_result.get("decision")
            decision_error_flag = bool("error" in decision_result or decision_result.get("status") == "error")

        self.state.rolling_confidence = float(
            0.7 * self.state.rolling_confidence + 0.3 * decision_confidence
        )

        uncertainty = float(np.clip(1.0 - decision_confidence, 0.0, 1.0))
        meta_uncertainty = None
        if isinstance(meta_analysis, dict):
            meta_uncertainty = meta_analysis.get("risk") or meta_analysis.get("uncertainty")
            if isinstance(meta_uncertainty, (int, float)):
                meta_uncertainty = float(np.clip(meta_uncertainty, 0.0, 1.0))
            else:
                meta_uncertainty = None

        error_probability = self._estimate_error_probability(
            decision_error_flag=decision_error_flag,
            reward=reward,
            meta_analysis=meta_analysis,
        )
        if error_probability >= self.config.anomaly_threshold:
            self.state.cumulative_errors += 1

        dominant_goal = self._extract_dominant_goal(goals, motivation)
        attention_summary = self._summarise_attention(attention_focus)
        belief_state = {
            "current_goal": dominant_goal,
            "last_decision": decision_choice,
            "confidence": decision_confidence,
            "rolling_confidence": self.state.rolling_confidence,
            "attention_focus": attention_summary,
            "motivation": self._normalise_motivation(motivation),
            "memory_load": self._estimate_memory_load(memory_result, working_memory),
        }

        emotion_summary = None
        if isinstance(emotion_state, dict):
            emotion_summary = {
                "valence": float(np.clip(emotion_state.get("valence", 0.0), -1.0, 1.0)),
                "arousal": float(np.clip(emotion_state.get("arousal", 0.0), 0.0, 1.0)),
                "dominance": float(np.clip(emotion_state.get("dominance", 0.5), 0.0, 1.0)),
            }
            belief_state["emotion"] = emotion_summary

        if isinstance(plan_result, dict):
            belief_state["plan_status"] = plan_result.get("status") or plan_result.get("summary")

        alerts: List[str] = []
        if uncertainty > (1.0 - self.config.min_confidence):
            alerts.append("high_uncertainty")
        if error_probability >= self.config.anomaly_threshold:
            alerts.append("risk_of_failure")
        if self.state.rolling_confidence < self.config.min_confidence:
            alerts.append("low_self_trust")

        introspection = {
            "uncertainty": uncertainty,
            "meta_uncertainty": meta_uncertainty,
            "error_probability": error_probability,
            "cumulative_errors": self.state.cumulative_errors,
            "observations": self.state.total_observations,
            "alerts": alerts,
        }

        attention_details = list(attention_scores or [])
        observation = {
            "time": time_point,
            "belief_state": belief_state,
            "introspection": introspection,
            "attention_details": attention_details,
        }
        self.observation_window.append(observation)
        self.state.belief_state = belief_state

        report_needed = (
            time_point - self.state.last_report_time >= self.config.introspection_interval
            or alerts
        )
        insight_text = None
        if report_needed and self.config.enable_text_report:
            insight_text = self._compose_text_report(belief_state, introspection)
            self.state.last_report_time = time_point

        report = {
            "time": time_point,
            "beliefs": belief_state,
            "introspection": introspection,
            "attention": {
                "focus": attention_summary,
                "details": attention_details[:3],
            },
            "alerts": alerts,
            "insight": insight_text,
        }
        self.reports.append(report)
        return report

    def get_recent_reports(self) -> List[Dict[str, Any]]:
        return list(self.reports)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "window_size": self.config.window_size,
                "report_history": self.config.report_history,
                "introspection_interval": self.config.introspection_interval,
                "min_confidence": self.config.min_confidence,
                "anomaly_threshold": self.config.anomaly_threshold,
                "enable_text_report": self.config.enable_text_report,
                "track_subsystems": list(self.config.track_subsystems),
            },
            "state": {
                "last_report_time": self.state.last_report_time,
                "cumulative_errors": self.state.cumulative_errors,
                "total_observations": self.state.total_observations,
                "rolling_confidence": self.state.rolling_confidence,
                "belief_state": self.state.belief_state,
            },
            "reports": list(self.reports),
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        state_data = data.get("state", {})
        self.state = SelfModelState(
            last_report_time=float(state_data.get("last_report_time", 0.0)),
            cumulative_errors=int(state_data.get("cumulative_errors", 0)),
            total_observations=int(state_data.get("total_observations", 0)),
            rolling_confidence=float(state_data.get("rolling_confidence", 0.5)),
            belief_state=state_data.get("belief_state", {}),
        )
        self.reports.clear()
        for report in data.get("reports", []):
            if isinstance(report, dict):
                self.reports.append(report)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _estimate_error_probability(
        *,
        decision_error_flag: bool,
        reward: Optional[float],
        meta_analysis: Optional[Dict[str, Any]],
    ) -> float:
        probability = 0.0
        if decision_error_flag:
            probability = max(probability, 0.75)
        if isinstance(reward, (int, float)) and reward < 0:
            probability = max(probability, float(np.clip(abs(reward), 0.0, 1.0)))
        if isinstance(meta_analysis, dict):
            if meta_analysis.get("error"):
                probability = max(probability, 0.9)
            meta_risk = meta_analysis.get("risk") or meta_analysis.get("uncertainty")
            if isinstance(meta_risk, (int, float)):
                probability = max(probability, float(np.clip(meta_risk, 0.0, 1.0)))
        return float(np.clip(probability, 0.0, 1.0))

    @staticmethod
    def _extract_dominant_goal(
        goals: Iterable[Any],
        motivation: Optional[Dict[str, float]],
    ) -> Optional[str]:
        dominant_goal = None
        max_motivation = -math.inf
        if isinstance(motivation, dict):
            for key, value in motivation.items():
                if isinstance(value, (int, float)) and float(value) > max_motivation:
                    dominant_goal = str(key)
                    max_motivation = float(value)
        if dominant_goal is None:
            for goal in goals or []:
                if goal:
                    dominant_goal = str(goal)
                    break
        return dominant_goal

    @staticmethod
    def _summarise_attention(
        focus: Optional[Iterable[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if not focus:
            return []
        summary: List[Dict[str, Any]] = []
        for item in list(focus)[:3]:
            if isinstance(item, dict):
                summary.append(
                    {
                        "source": str(item.get("source", "unknown")),
                        "score": float(item.get("score", 0.0)) if isinstance(item.get("score"), (int, float)) else 0.0,
                    }
                )
        return summary

    @staticmethod
    def _normalise_motivation(motivation: Optional[Dict[str, float]]) -> Dict[str, float]:
        if not isinstance(motivation, dict):
            return {}
        result: Dict[str, float] = {}
        for key, value in motivation.items():
            if isinstance(value, (int, float)):
                result[str(key)] = float(np.clip(value, 0.0, 1.0))
        return result

    @staticmethod
    def _estimate_memory_load(
        memory_result: Optional[Dict[str, Any]],
        working_memory: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        stats = {}
        if isinstance(memory_result, dict):
            statistics = memory_result.get("statistics")
            if isinstance(statistics, dict):
                stats.update(
                    {
                        "traces": statistics.get("total_traces"),
                        "working_items": statistics.get("working_memory_items"),
                    }
                )
        if isinstance(working_memory, dict):
            stats.setdefault("working_depth", len(working_memory.get("items", [])))
        return stats

    def _compose_text_report(
        self,
        belief_state: Dict[str, Any],
        introspection: Dict[str, Any],
    ) -> str:
        focus_sources = ", ".join(
            f"{item['source']}({item['score']:.2f})"
            for item in belief_state.get("attention_focus", [])
        ) or "none"
        alerts = introspection.get("alerts", [])
        alert_text = ", ".join(alerts) if alerts else "stable"
        goal_text = belief_state.get("current_goal", "unspecified")
        confidence = belief_state.get("rolling_confidence", 0.5)
        uncertainty = introspection.get("uncertainty", 0.5)

        lines = [
            f"Goal focus: {goal_text}.",
            f"Confidence={confidence:.2f}, Uncertainty={uncertainty:.2f}.",
            f"Attention: {focus_sources}.",
            f"System alerts: {alert_text}.",
        ]
        emotion = belief_state.get("emotion")
        if isinstance(emotion, dict):
            lines.append(
                f"Valence={emotion.get('valence', 0.0):.2f}, Arousal={emotion.get('arousal', 0.0):.2f}."
            )
        motivation = belief_state.get("motivation", {})
        if motivation:
            top_motives = sorted(motivation.items(), key=lambda it: it[1], reverse=True)[:2]
            motive_text = ", ".join(f"{k}:{v:.2f}" for k, v in top_motives)
            lines.append(f"Motivation snapshot: {motive_text}.")
        return " ".join(lines)
