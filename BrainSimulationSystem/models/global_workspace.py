"""Global workspace + metacognitive control module.

This module provides a lightweight implementation of a Baars-style Global
Workspace that collects salient items from cognitive subsystems and broadcasts a
small set of "winning" items. A companion metacognitive controller translates
workspace signals (uncertainty, prediction errors, threat) into control
commands, such as:

- attention directive overrides (top-down bias for next step)
- learning-rate scaling (plasticity gate)
- decision exploration adjustments (via DecisionProcess meta hooks)

The implementation is intentionally dependency-light and heuristic so it can be
plugged into ``BrainSimulationSystem.brain_simulation.BrainSimulation`` without
requiring external NLP/ML libraries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple
from collections import deque
import time

import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(float(value), float(lo), float(hi)))


def _normalise_terms(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        token = value.strip()
        return [token] if token else []
    if isinstance(value, (list, tuple, set)):
        terms: List[str] = []
        for item in value:
            if item is None:
                continue
            token = str(item).strip()
            if token and token.lower() not in (t.lower() for t in terms):
                terms.append(token)
        return terms
    token = str(value).strip()
    return [token] if token else []


@dataclass
class GlobalWorkspaceConfig:
    enabled: bool = True
    broadcast_capacity: int = 6
    max_items: int = 48
    decay: float = 0.92
    min_salience: float = 0.05

    low_confidence_threshold: float = 0.35
    low_reliability_threshold: float = 0.45
    threat_suppress_threshold: float = 0.65
    rpe_boost_threshold: float = 0.25

    rpe_learning_gain: float = 1.4
    prediction_error_gain: float = 0.8
    threat_learning_suppress: float = 0.7
    learning_rate_min_scale: float = 0.2
    learning_rate_max_scale: float = 3.0

    attention_override_hold_steps: int = 2
    focus_term_limit: int = 6
    workspace_focus_cap: int = 4

    exploration_delta_on_uncertainty: float = 0.05
    exploration_delta_on_threat: float = -0.06


@dataclass
class WorkspaceItem:
    """Candidate item competing for global broadcast."""

    type: str
    source: str
    summary: str
    salience: float
    payload: Dict[str, Any] = field(default_factory=dict)
    tags: Tuple[str, ...] = field(default_factory=tuple)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": self.type,
            "source": self.source,
            "summary": self.summary,
            "salience": float(self.salience),
            "timestamp": float(self.timestamp),
            "payload": dict(self.payload),
        }
        if self.tags:
            data["tags"] = list(self.tags)
        return data


class GlobalWorkspace:
    """Maintain a small set of salient items and produce broadcasts."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = GlobalWorkspaceConfig(**(config or {}))
        self.items: Deque[WorkspaceItem] = deque(maxlen=max(1, int(self.config.max_items)))
        self.last_broadcast: List[Dict[str, Any]] = []
        self._sequence = 0

    def reset(self) -> None:
        self.items.clear()
        self.last_broadcast = []
        self._sequence = 0

    def add_items(self, items: Iterable[WorkspaceItem]) -> None:
        if not self.config.enabled:
            return
        self._decay_existing()
        for item in items:
            if item is None:
                continue
            salience = _clip(_safe_float(item.salience, 0.0), 0.0, 1.0)
            if salience < float(self.config.min_salience):
                continue
            item.salience = salience
            self.items.append(item)
        self._prune()

    def broadcast(self) -> Dict[str, Any]:
        if not self.config.enabled:
            return {"enabled": False, "broadcast": [], "items": 0, "sequence": self._sequence}

        self._decay_existing()
        self._prune()
        ranked = sorted(self.items, key=lambda item: item.salience, reverse=True)
        capacity = max(1, int(self.config.broadcast_capacity))
        broadcast_items = ranked[:capacity]
        payload = [item.to_dict() for item in broadcast_items]
        self.last_broadcast = payload
        self._sequence += 1
        return {
            "enabled": True,
            "broadcast": payload,
            "dominant": payload[0] if payload else None,
            "items": len(self.items),
            "sequence": self._sequence,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {k: getattr(self.config, k) for k in self.config.__dataclass_fields__},
            "sequence": self._sequence,
            "items": [item.to_dict() for item in list(self.items)],
            "last_broadcast": list(self.last_broadcast),
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        cfg = data.get("config")
        if isinstance(cfg, dict):
            self.config = GlobalWorkspaceConfig(**cfg)
            self.items = deque(maxlen=max(1, int(self.config.max_items)))
        self._sequence = int(data.get("sequence", 0))
        self.last_broadcast = list(data.get("last_broadcast", []))
        self.items.clear()
        for raw in data.get("items", []):
            if not isinstance(raw, dict):
                continue
            item = WorkspaceItem(
                type=str(raw.get("type", "unknown")),
                source=str(raw.get("source", "unknown")),
                summary=str(raw.get("summary", "")),
                salience=_clip(_safe_float(raw.get("salience"), 0.0), 0.0, 1.0),
                payload=dict(raw.get("payload") or {}),
                tags=tuple(str(tag) for tag in raw.get("tags", []) if tag is not None),
                timestamp=_safe_float(raw.get("timestamp"), time.time()),
            )
            self.items.append(item)
        self._prune()

    # ------------------------------------------------------------------ #
    def _decay_existing(self) -> None:
        decay = _clip(_safe_float(self.config.decay, 0.92), 0.0, 0.9999)
        if decay >= 0.9999:
            return
        for item in list(self.items):
            item.salience *= decay

    def _prune(self) -> None:
        threshold = float(self.config.min_salience)
        kept = [item for item in list(self.items) if float(item.salience) >= threshold]
        kept.sort(key=lambda item: item.timestamp, reverse=True)
        self.items.clear()
        for item in kept[: max(1, int(self.config.max_items))]:
            self.items.append(item)


class MetacognitiveController:
    """Convert global workspace signals into control commands."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.workspace = GlobalWorkspace(config or {})
        self.config = self.workspace.config
        self.last_commands: Dict[str, Any] = {}
        self.pending_attention_overrides: Dict[str, Any] = {}
        self.pending_attention_steps: int = 0
        self._pending_decision_adjustments: Dict[str, float] = {}

    def reset(self) -> None:
        self.workspace.reset()
        self.last_commands = {}
        self.pending_attention_overrides = {}
        self.pending_attention_steps = 0
        self._pending_decision_adjustments = {}

    def consume_pending_adjustments(self, stage: str) -> Dict[str, float]:
        """DecisionProcess meta hook: return (and clear) pending deltas."""

        if stage != "process":
            return {}
        adjustments = dict(self._pending_decision_adjustments)
        self._pending_decision_adjustments = {}
        return adjustments

    def update(
        self,
        *,
        time_point: float,
        decision_result: Optional[Dict[str, Any]] = None,
        emotion_result: Optional[Dict[str, Any]] = None,
        meta_analysis: Optional[Dict[str, Any]] = None,
        self_model: Optional[Dict[str, Any]] = None,
        attention_focus_bundle: Optional[Dict[str, Any]] = None,
        memory_result: Optional[Dict[str, Any]] = None,
        language_context: Optional[Dict[str, Any]] = None,
        self_supervised_summary: Optional[Dict[str, Any]] = None,
        plan_result: Optional[Dict[str, Any]] = None,
        decision_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.config.enabled:
            return {"enabled": False}

        now = float(time_point)
        candidates = self._build_candidates(
            now=now,
            decision_result=decision_result,
            emotion_result=emotion_result,
            meta_analysis=meta_analysis,
            self_model=self_model,
            attention_focus_bundle=attention_focus_bundle,
            memory_result=memory_result,
            language_context=language_context,
            self_supervised_summary=self_supervised_summary,
            plan_result=plan_result,
        )
        self.workspace.add_items(candidates)
        workspace_snapshot = self.workspace.broadcast()

        signals = self._extract_signals(
            decision_result=decision_result,
            emotion_result=emotion_result,
            meta_analysis=meta_analysis,
            self_model=self_model,
            self_supervised_summary=self_supervised_summary,
        )
        commands = self._compute_commands(
            signals=signals,
            decision_context=decision_context or {},
            language_context=language_context or {},
        )
        self.last_commands = commands

        attention_overrides = commands.get("attention_overrides")
        if isinstance(attention_overrides, dict) and attention_overrides:
            self.pending_attention_overrides = attention_overrides
            self.pending_attention_steps = int(commands.get("attention_hold_steps") or self.config.attention_override_hold_steps)

        decision_adjustments = commands.get("decision_adjustments")
        if isinstance(decision_adjustments, dict) and decision_adjustments:
            cleaned: Dict[str, float] = {}
            for key, value in decision_adjustments.items():
                if isinstance(value, (int, float)) and abs(float(value)) > 1e-9:
                    cleaned[str(key)] = float(value)
            if cleaned:
                self._pending_decision_adjustments.update(cleaned)

        return {
            "enabled": True,
            "time": now,
            "workspace": workspace_snapshot,
            "signals": signals,
            "commands": commands,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace": self.workspace.to_dict(),
            "last_commands": dict(self.last_commands),
            "pending_attention_overrides": dict(self.pending_attention_overrides),
            "pending_attention_steps": int(self.pending_attention_steps),
            "pending_decision_adjustments": dict(self._pending_decision_adjustments),
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        ws = data.get("workspace")
        if isinstance(ws, dict):
            self.workspace.from_dict(ws)
            self.config = self.workspace.config
        self.last_commands = dict(data.get("last_commands") or {})
        self.pending_attention_overrides = dict(data.get("pending_attention_overrides") or {})
        self.pending_attention_steps = int(data.get("pending_attention_steps", 0))
        pending = data.get("pending_decision_adjustments") or {}
        if isinstance(pending, dict):
            self._pending_decision_adjustments = {
                str(k): float(v) for k, v in pending.items() if isinstance(v, (int, float))
            }
        else:
            self._pending_decision_adjustments = {}

    # ------------------------------------------------------------------ #
    def _build_candidates(
        self,
        *,
        now: float,
        decision_result: Optional[Dict[str, Any]],
        emotion_result: Optional[Dict[str, Any]],
        meta_analysis: Optional[Dict[str, Any]],
        self_model: Optional[Dict[str, Any]],
        attention_focus_bundle: Optional[Dict[str, Any]],
        memory_result: Optional[Dict[str, Any]],
        language_context: Optional[Dict[str, Any]],
        self_supervised_summary: Optional[Dict[str, Any]],
        plan_result: Optional[Dict[str, Any]],
    ) -> List[WorkspaceItem]:
        items: List[WorkspaceItem] = []

        if isinstance(decision_result, dict):
            conf = _clip(_safe_float(decision_result.get("confidence"), 0.0), 0.0, 1.0)
            choice = decision_result.get("decision")
            if choice is not None:
                items.append(
                    WorkspaceItem(
                        type="decision",
                        source="decision",
                        summary=f"decision={choice} confidence={conf:.2f}",
                        salience=_clip(1.0 - conf, 0.0, 1.0),
                        payload={"decision": choice, "confidence": conf},
                        tags=("uncertainty",) if conf < 0.5 else tuple(),
                        timestamp=now,
                    )
                )

        threat = self._extract_threat(emotion_result)
        if threat > 0.0:
            items.append(
                WorkspaceItem(
                    type="threat",
                    source="limbic",
                    summary=f"threat_level={threat:.2f}",
                    salience=_clip(threat, 0.0, 1.0),
                    payload={"threat_level": threat},
                    tags=("threat", "arousal"),
                    timestamp=now,
                )
            )

        rpe = _safe_float((emotion_result or {}).get("reward_prediction_error"), 0.0) if isinstance(emotion_result, dict) else 0.0
        if abs(rpe) > 1e-9:
            magnitude = _clip(abs(rpe), 0.0, 1.0)
            items.append(
                WorkspaceItem(
                    type="reward_prediction_error",
                    source="limbic",
                    summary=f"rpe={rpe:.2f}",
                    salience=magnitude,
                    payload={"rpe": float(rpe)},
                    tags=("rpe", "learning"),
                    timestamp=now,
                )
            )

        pred_err = None
        if isinstance(self_supervised_summary, dict):
            pred_err = self_supervised_summary.get("prediction_error")
        if isinstance(pred_err, (int, float)) and abs(float(pred_err)) > 1e-9:
            magnitude = _clip(abs(_safe_float(pred_err, 0.0)), 0.0, 1.0)
            items.append(
                WorkspaceItem(
                    type="prediction_error",
                    source="self_supervised",
                    summary=f"prediction_error={float(pred_err):.3f}",
                    salience=magnitude,
                    payload={"prediction_error": float(pred_err)},
                    tags=("prediction_error", "learning"),
                    timestamp=now,
                )
            )

        if isinstance(meta_analysis, dict):
            reliability = meta_analysis.get("reliability_score")
            if isinstance(reliability, (int, float)):
                reliability_f = _clip(_safe_float(reliability, 0.5), 0.0, 1.0)
                items.append(
                    WorkspaceItem(
                        type="meta_reliability",
                        source="meta_reasoner",
                        summary=f"reliability={reliability_f:.2f}",
                        salience=_clip(1.0 - reliability_f, 0.0, 1.0),
                        payload={"reliability_score": reliability_f},
                        tags=("reliability", "uncertainty"),
                        timestamp=now,
                    )
                )

        if isinstance(self_model, dict):
            alerts = self_model.get("alerts")
            if isinstance(alerts, list):
                for alert in alerts[:4]:
                    if alert is None:
                        continue
                    token = str(alert).strip()
                    if not token:
                        continue
                    items.append(
                        WorkspaceItem(
                            type="self_alert",
                            source="self_model",
                            summary=token,
                            salience=0.65,
                            payload={"alert": token},
                            tags=("self_monitoring",),
                            timestamp=now,
                        )
                    )

        if isinstance(attention_focus_bundle, dict):
            focus = attention_focus_bundle.get("focus")
            if isinstance(focus, list):
                for entry in focus[:3]:
                    if not isinstance(entry, dict):
                        continue
                    score = _clip(_safe_float(entry.get("score"), 0.0), 0.0, 1.0)
                    source = str(entry.get("source", "workspace"))
                    payload = entry.get("payload")
                    summary = None
                    if isinstance(payload, dict):
                        summary = payload.get("type") or payload.get("summary")
                    if not summary:
                        summary = source
                    items.append(
                        WorkspaceItem(
                            type="attention_focus",
                            source=source,
                            summary=str(summary),
                            salience=_clip(score, 0.0, 1.0),
                            payload={"score": score, "payload": payload},
                            tags=("attention",),
                            timestamp=now,
                        )
                    )

        if isinstance(memory_result, dict):
            retrieved = memory_result.get("retrieved")
            if isinstance(retrieved, list):
                for entry in retrieved[:2]:
                    if not isinstance(entry, dict):
                        continue
                    score = entry.get("score")
                    score_f = _clip(_safe_float(score, 0.0), 0.0, 1.0) if isinstance(score, (int, float)) else 0.4
                    content = entry.get("content")
                    concept = None
                    if isinstance(content, dict):
                        concept = content.get("concept") or content.get("event")
                    if not concept:
                        concept = str(entry.get("id") or "memory")
                    items.append(
                        WorkspaceItem(
                            type="memory_retrieval",
                            source="memory",
                            summary=str(concept),
                            salience=_clip(score_f, 0.0, 1.0),
                            payload={"score": score_f, "content": content},
                            tags=("memory",),
                            timestamp=now,
                        )
                    )

        if isinstance(plan_result, dict):
            next_action = plan_result.get("next_action") or plan_result.get("plan_next_action")
            if next_action is None:
                seq = plan_result.get("sequence")
                if isinstance(seq, list) and seq:
                    next_action = seq[0]
            if next_action is not None:
                token = str(next_action).strip()
                if token:
                    items.append(
                        WorkspaceItem(
                            type="plan_next_action",
                            source="planner",
                            summary=token,
                            salience=0.4,
                            payload={"next_action": token},
                            tags=("plan",),
                            timestamp=now,
                        )
                    )

        if isinstance(language_context, dict):
            summary = language_context.get("last_summary") or language_context.get("summary")
            if isinstance(summary, str) and summary.strip():
                items.append(
                    WorkspaceItem(
                        type="language_summary",
                        source="language",
                        summary=summary.strip()[:160],
                        salience=0.35,
                        payload={"summary": summary.strip()},
                        tags=("language",),
                        timestamp=now,
                    )
                )

        return items

    @staticmethod
    def _extract_threat(emotion_result: Optional[Dict[str, Any]]) -> float:
        if not isinstance(emotion_result, dict):
            return 0.0
        limbic = emotion_result.get("limbic_circuits")
        if isinstance(limbic, dict):
            amygdala = limbic.get("amygdala")
            if isinstance(amygdala, dict):
                if amygdala.get("threat_level") is not None:
                    return _clip(_safe_float(amygdala.get("threat_level"), 0.0), 0.0, 1.0)
        if emotion_result.get("threat") is not None:
            return _clip(_safe_float(emotion_result.get("threat"), 0.0), 0.0, 1.0)
        return 0.0

    def _extract_signals(
        self,
        *,
        decision_result: Optional[Dict[str, Any]],
        emotion_result: Optional[Dict[str, Any]],
        meta_analysis: Optional[Dict[str, Any]],
        self_model: Optional[Dict[str, Any]],
        self_supervised_summary: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        confidence = _clip(_safe_float((decision_result or {}).get("confidence"), 0.0), 0.0, 1.0) if isinstance(decision_result, dict) else 0.0
        uncertainty = _clip(1.0 - confidence, 0.0, 1.0)
        rpe = _safe_float((emotion_result or {}).get("reward_prediction_error"), 0.0) if isinstance(emotion_result, dict) else 0.0
        threat = self._extract_threat(emotion_result)
        reliability = None
        if isinstance(meta_analysis, dict) and isinstance(meta_analysis.get("reliability_score"), (int, float)):
            reliability = _clip(_safe_float(meta_analysis.get("reliability_score"), 0.5), 0.0, 1.0)
        prediction_error = None
        if isinstance(self_supervised_summary, dict) and isinstance(self_supervised_summary.get("prediction_error"), (int, float)):
            prediction_error = _safe_float(self_supervised_summary.get("prediction_error"), 0.0)

        alerts: List[str] = []
        if isinstance(self_model, dict):
            raw_alerts = self_model.get("alerts")
            if isinstance(raw_alerts, list):
                alerts = [str(a) for a in raw_alerts if a is not None]

        return {
            "decision_confidence": confidence,
            "decision_uncertainty": uncertainty,
            "reward_prediction_error": float(rpe),
            "rpe_magnitude": float(abs(rpe)),
            "threat_level": threat,
            "reliability_score": reliability,
            "prediction_error": prediction_error,
            "self_alerts": alerts,
        }

    def _compute_commands(
        self,
        *,
        signals: Dict[str, Any],
        decision_context: Dict[str, Any],
        language_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        confidence = _safe_float(signals.get("decision_confidence"), 0.0)
        uncertainty = _safe_float(signals.get("decision_uncertainty"), 1.0)
        threat = _safe_float(signals.get("threat_level"), 0.0)
        rpe_mag = _safe_float(signals.get("rpe_magnitude"), 0.0)
        reliability = signals.get("reliability_score")
        reliability_val = _safe_float(reliability, 0.5) if reliability is not None else None
        prediction_error = signals.get("prediction_error")
        pred_mag = _safe_float(prediction_error, 0.0) if prediction_error is not None else 0.0
        alerts = signals.get("self_alerts") or []

        request_more = False
        notes: List[str] = []

        if confidence < float(self.config.low_confidence_threshold) or "high_uncertainty" in alerts:
            request_more = True
            notes.append("low_decision_confidence")
        if reliability_val is not None and reliability_val < float(self.config.low_reliability_threshold):
            request_more = True
            notes.append("low_meta_reliability")

        learning_scale = 1.0
        if rpe_mag >= float(self.config.rpe_boost_threshold):
            learning_scale *= 1.0 + float(self.config.rpe_learning_gain) * _clip(rpe_mag, 0.0, 1.0)
            notes.append("rpe_plasticity_boost")
        if pred_mag > 1e-9:
            learning_scale *= 1.0 + float(self.config.prediction_error_gain) * _clip(pred_mag, 0.0, 1.0)
            notes.append("prediction_error_boost")
        if threat >= float(self.config.threat_suppress_threshold):
            learning_scale *= float(self.config.threat_learning_suppress)
            notes.append("threat_plasticity_suppress")

        learning_scale = _clip(
            learning_scale,
            float(self.config.learning_rate_min_scale),
            float(self.config.learning_rate_max_scale),
        )

        attention_overrides: Dict[str, Any] = {}
        modality_weights: Dict[str, float] = {}
        workspace_attention: Dict[str, Any] = {}

        if request_more:
            modality_weights["language"] = 1.0
            modality_weights["structured"] = 0.75
            workspace_attention["priority"] = 0.9

        if threat > 0.0:
            modality_weights["somatosensory"] = _clip(0.6 + 0.4 * threat, 0.0, 1.0)

        focus_terms: List[str] = []
        focus_terms.extend(_normalise_terms(decision_context.get("language_key_terms")))
        focus_terms.extend(_normalise_terms(decision_context.get("language_action_items")))
        focus_terms.extend(_normalise_terms(language_context.get("workspace_focus")))
        focus_terms.extend(_normalise_terms(language_context.get("semantic_focus")))
        focus_terms = [t for t in focus_terms if t]

        # Deduplicate preserving order (case-insensitive).
        unique_terms: List[str] = []
        seen = set()
        for term in focus_terms:
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_terms.append(term)
            if len(unique_terms) >= int(self.config.focus_term_limit):
                break

        if unique_terms:
            attention_overrides["semantic_focus"] = unique_terms
            attention_overrides["workspace_focus"] = unique_terms[: int(self.config.workspace_focus_cap)]

        if modality_weights:
            attention_overrides["modality_weights"] = modality_weights
        if workspace_attention:
            attention_overrides["workspace_attention"] = workspace_attention

        decision_adjustments: Dict[str, float] = {}
        exploration_delta = 0.0
        if request_more:
            exploration_delta += float(self.config.exploration_delta_on_uncertainty)
        if threat >= float(self.config.threat_suppress_threshold):
            exploration_delta += float(self.config.exploration_delta_on_threat)
        if abs(exploration_delta) > 1e-9:
            decision_adjustments["exploration_rate"] = float(exploration_delta)

        return {
            "request_more_information": bool(request_more),
            "learning_rate_scale": float(learning_scale),
            "attention_overrides": attention_overrides,
            "attention_hold_steps": int(self.config.attention_override_hold_steps),
            "decision_adjustments": decision_adjustments,
            "notes": notes,
            "signals": {
                "uncertainty": float(_clip(uncertainty, 0.0, 1.0)),
                "threat_level": float(_clip(threat, 0.0, 1.0)),
                "rpe_magnitude": float(rpe_mag),
                "prediction_error": float(pred_mag) if prediction_error is not None else None,
            },
        }


__all__ = ["GlobalWorkspace", "MetacognitiveController", "WorkspaceItem"]

