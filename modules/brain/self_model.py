from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from backend.autogpt.autogpt.core.knowledge_graph.graph_store import GraphStore
from backend.autogpt.autogpt.core.knowledge_graph.ontology import EntityType
from backend.knowledge.registry import get_graph_store_instance


logger = logging.getLogger(__name__)


def _utc_timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class StrategyStats:
    """Track performance metrics for a named strategy."""

    name: str
    attempts: int = 0
    successes: int = 0
    last_confidence: Optional[float] = None
    last_outcome: Optional[str] = None
    last_used: Optional[str] = None

    def register(self, success: Optional[bool], confidence: Optional[float]) -> None:
        self.attempts += 1
        if success:
            self.successes += 1
            self.last_outcome = "success"
        elif success is False:
            self.last_outcome = "failure"
        else:
            self.last_outcome = "unknown"
        if confidence is not None:
            self.last_confidence = max(0.0, min(1.0, confidence))
        self.last_used = _utc_timestamp()

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "attempts": self.attempts,
            "successes": self.successes,
            "success_rate": round(self.success_rate, 3),
            "last_outcome": self.last_outcome,
            "last_used": self.last_used,
        }
        if self.last_confidence is not None:
            payload["last_confidence"] = round(self.last_confidence, 3)
        return payload


@dataclass
class CapabilityStats:
    """Adaptive confidence profile for a specific capability."""

    name: str
    display_name: str
    weight: float = 0.5
    successes: int = 0
    failures: int = 0
    total: int = 0
    streak: int = 0
    last_outcome: Optional[str] = None
    last_used: Optional[str] = None

    def register(self, success: Optional[bool], influence: Optional[float]) -> None:
        """Update running weight and counters based on outcome."""

        influence_val = _safe_float(influence) or 0.5
        influence_val = max(0.05, min(1.0, influence_val))
        timestamp = _utc_timestamp()
        self.total += 1
        self.last_used = timestamp

        base_step = 0.08 * influence_val
        if success is True:
            self.successes += 1
            self.streak = self.streak + 1 if self.streak >= 0 else 1
            bonus = min(0.05, 0.01 * abs(self.streak))
            self.weight = min(1.0, self.weight + base_step + bonus)
            self.last_outcome = "success"
        elif success is False:
            self.failures += 1
            self.streak = self.streak - 1 if self.streak <= 0 else -1
            penalty = min(0.06, 0.012 * abs(self.streak))
            self.weight = max(0.0, self.weight - (base_step * 1.2 + penalty))
            self.last_outcome = "failure"
        else:
            drift = 0.02 * influence_val
            self.weight = max(0.0, min(1.0, self.weight + (drift if self.weight < 0.5 else -drift)))
            self.last_outcome = "unknown"

    @property
    def success_rate(self) -> float:
        return self.successes / self.total if self.total else 0.0

    @property
    def needs_improvement(self) -> bool:
        return self.total >= 3 and self.success_rate < 0.4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "weight": round(self.weight, 3),
            "success_rate": round(self.success_rate, 3),
            "attempts": self.total,
            "successes": self.successes,
            "failures": self.failures,
            "last_outcome": self.last_outcome,
            "last_used": self.last_used,
        }


@dataclass
class SelfModelState:
    """Internal representation of self-awareness signals."""

    current_goal: Optional[str] = None
    assumptions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    strategies: Dict[str, StrategyStats] = field(default_factory=dict)
    cycle_history: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=32))
    total_attempts: int = 0
    total_successes: int = 0
    capabilities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_task: Dict[str, Any] = field(default_factory=dict)
    weaknesses: Dict[str, float] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.total_successes / self.total_attempts if self.total_attempts else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "current_goal": self.current_goal,
            "assumptions": dict(self.assumptions),
            "strategies": [stats.as_dict() for stats in self.strategies.values()],
            "strategies_tried": list(self.strategies.keys()),
            "success_rate": round(self.success_rate, 3),
            "total_attempts": self.total_attempts,
            "total_successes": self.total_successes,
            "capabilities": dict(self.capabilities),
            "last_task_performance": dict(self.last_task),
            "history": list(self.cycle_history),
            "weaknesses": dict(self.weaknesses),
        }


class SelfModel:
    """Maintain a self-representation and synchronise it with the knowledge graph."""

    def __init__(
        self,
        *,
        node_id: str = "SELF",
        graph_store: GraphStore | None = None,
        history_limit: int = 32,
        capability_defaults: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.node_id = node_id
        self._graph = graph_store or get_graph_store_instance()
        self.state = SelfModelState()
        self.state.cycle_history = deque(maxlen=max(8, history_limit))
        self._capability_profiles: Dict[str, CapabilityStats] = {}
        self._default_capability_weight = 0.5
        if capability_defaults:
            for name, weight in capability_defaults.items():
                self._register_capability_seed(name, _safe_float(weight))
        self._ensure_self_node()
        self._refresh_capabilities()

    # ------------------------------------------------------------------ #
    def record_cycle(
        self,
        *,
        goal: Optional[str],
        assumptions: Optional[Mapping[str, Any] | Iterable[str]] = None,
        strategies: Optional[Mapping[str, Any]] = None,
        executed_strategy: Optional[str] = None,
        success: Optional[bool] = None,
        reward: Optional[float] = None,
        feedback: Optional[Mapping[str, Any]] = None,
        context: Optional[Mapping[str, Any]] = None,
        errors: Optional[Sequence[str]] = None,
        capabilities: Optional[Mapping[str, Any] | Sequence[str]] = None,
        emotion: Optional[Mapping[str, Any]] = None,
        curiosity_drive: Optional[float] = None,
    ) -> None:
        """Update internal state based on the latest cognitive cycle."""

        timestamp = _utc_timestamp()
        if goal:
            self.state.current_goal = str(goal)

        if assumptions:
            for key, value in self._normalise_assumptions(assumptions).items():
                self.state.assumptions[key] = value

        score_map = self._normalise_scores(strategies)

        if score_map:
            for name, score in score_map.items():
                stats = self.state.strategies.setdefault(name, StrategyStats(name=name))
                stats.last_confidence = score

        strategy_name = executed_strategy or self._select_primary_strategy(score_map)
        if strategy_name:
            stats = self.state.strategies.setdefault(strategy_name, StrategyStats(name=strategy_name))
            stats.register(success, score_map.get(strategy_name))

        if success is not None:
            self.state.total_attempts += 1
            if success:
                self.state.total_successes += 1

        history_entry: Dict[str, Any] = {
            "timestamp": timestamp,
            "goal": self.state.current_goal,
            "strategy": strategy_name,
            "success": None if success is None else bool(success),
        }

        if curiosity_drive is not None:
            curiosity_value = _safe_float(curiosity_drive)
            if curiosity_value is not None:
                history_entry["curiosity"] = curiosity_value

        if score_map and strategy_name in score_map:
            history_entry["confidence"] = round(score_map[strategy_name], 3)

        reward_value = _safe_float(reward)
        if reward_value is not None:
            history_entry["reward"] = reward_value

        if feedback:
            history_entry["feedback"] = {
                key: val
                for key, raw in feedback.items()
                if (val := _safe_float(raw)) is not None
            }

        if errors:
            filtered_errors = [str(err) for err in errors if err]
            if filtered_errors:
                history_entry["errors"] = list(dict.fromkeys(filtered_errors))

        emotion_snapshot = self._normalise_emotion(emotion)
        if emotion_snapshot:
            history_entry["emotion"] = emotion_snapshot

        context_snapshot = self._filter_context(context)
        if context_snapshot:
            history_entry["context"] = context_snapshot

        capability_signals = self._normalise_capability_signals(capabilities)
        if capability_signals:
            history_entry["capabilities"] = {}
            for norm_key, (display_name, influence) in capability_signals.items():
                history_entry["capabilities"][display_name] = round(float(influence), 3)
                self._register_capability_event(display_name, influence, success, norm_key=norm_key)

        if self.state.strategies:
            history_entry["strategies"] = {
                name: stats.as_dict()
                for name, stats in list(self.state.strategies.items())[:3]
            }

        self.state.cycle_history.append(history_entry)
        self._refresh_capabilities()
        self.state.last_task = dict(history_entry)
        self._persist(timestamp)

    # ------------------------------------------------------------------ #
    def snapshot(self) -> Dict[str, Any]:
        """Return a serialisable view of the current self-model state."""

        return self.state.as_dict()

    # ------------------------------------------------------------------ #
    def _refresh_capabilities(self) -> None:
        capability_summary: Dict[str, Dict[str, Any]] = {}
        weaknesses: Dict[str, float] = {}

        for stats in self._capability_profiles.values():
            capability_summary[stats.display_name] = stats.to_dict()
            if stats.needs_improvement:
                weaknesses[f"capability:{stats.display_name}"] = round(stats.success_rate, 3)

        for name, stats in self.state.strategies.items():
            if stats.attempts >= 3 and stats.success_rate < 0.4:
                weaknesses[f"strategy:{name}"] = round(stats.success_rate, 3)

        self.state.capabilities = capability_summary
        self.state.weaknesses = weaknesses

    def capability_summary(self) -> Dict[str, Dict[str, Any]]:
        """Expose the current capability confidence table."""

        return dict(self.state.capabilities)

    def capability_weight(self, name: str, default: float = 0.5) -> float:
        """Return the learned weight for ``name`` (0.0-1.0)."""

        stats = self._capability_profiles.get(self._normalise_capability_name(name))
        if stats is None:
            return default
        return max(0.0, min(1.0, float(stats.weight)))

    def _register_capability_seed(self, name: str, weight: Optional[float]) -> None:
        norm = self._normalise_capability_name(name)
        if not norm:
            return
        initial = _safe_float(weight)
        if initial is None:
            initial = self._default_capability_weight
        initial = max(0.0, min(1.0, initial))
        self._capability_profiles[norm] = CapabilityStats(
            name=norm,
            display_name=str(name),
            weight=initial,
        )

    def _register_capability_event(
        self,
        name: str,
        influence: Optional[float],
        success: Optional[bool],
        *,
        norm_key: Optional[str] = None,
    ) -> None:
        norm = norm_key or self._normalise_capability_name(name)
        if not norm:
            return
        stats = self._capability_profiles.get(norm)
        if stats is None:
            stats = CapabilityStats(
                name=norm,
                display_name=str(name),
                weight=self._default_capability_weight,
            )
            self._capability_profiles[norm] = stats
        else:
            stats.display_name = str(name) or stats.display_name
        stats.register(success, influence)

    def _normalise_capability_signals(
        self,
        capabilities: Optional[Mapping[str, Any] | Sequence[str]],
    ) -> Dict[str, Tuple[str, float]]:
        if not capabilities:
            return {}

        signals: Dict[str, Tuple[str, float]] = {}
        if isinstance(capabilities, Mapping):
            iterable = capabilities.items()
        else:
            iterable = ((str(item), 1.0) for item in capabilities)

        for name, raw_value in iterable:
            norm = self._normalise_capability_name(name)
            if not norm:
                continue
            influence = _safe_float(raw_value)
            if influence is None:
                influence = self._default_capability_weight
            influence = max(0.0, min(1.0, influence))
            current = signals.get(norm)
            if current is None or influence > current[1]:
                signals[norm] = (str(name), influence)
        return signals

    @staticmethod
    def _normalise_capability_name(name: Any) -> str:
        if name is None:
            return ""
        text = str(name).strip().lower()
        return text

    def _persist(self, timestamp: Optional[str] = None) -> None:
        properties = self.state.as_dict()
        properties["updated_at"] = timestamp or _utc_timestamp()
        try:
            self._graph.add_node(self.node_id, EntityType.AGENT, **properties)
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Failed to persist self-model state to knowledge graph.", exc_info=True)

    def _ensure_self_node(self) -> None:
        try:
            snapshot = self._graph.query(node_id=self.node_id)
        except Exception:
            logger.debug("Knowledge graph query failed during self-node check.", exc_info=True)
            return
        nodes = snapshot.get("nodes", []) if isinstance(snapshot, dict) else []
        if any(getattr(node, "id", None) == self.node_id for node in nodes):
            return
        try:
            self._graph.add_node(self.node_id, EntityType.AGENT, role="self_model")
        except Exception:
            logger.debug("Failed to create SELF node in knowledge graph.", exc_info=True)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalise_assumptions(
        assumptions: Mapping[str, Any] | Iterable[str]
    ) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        if isinstance(assumptions, Mapping):
            for key, value in assumptions.items():
                value_dict = {"value": value}
                if isinstance(value, Mapping):
                    value_dict = dict(value)
                result[str(key)] = value_dict
        else:
            for item in assumptions:
                result[str(item)] = {"value": True}
        return result

    @staticmethod
    def _normalise_scores(scores: Optional[Mapping[str, Any]]) -> Dict[str, float]:
        result: Dict[str, float] = {}
        if not isinstance(scores, Mapping):
            return result
        for key, value in scores.items():
            val = _safe_float(value)
            if val is not None:
                result[str(key)] = max(0.0, min(1.0, val))
        return result

    @staticmethod
    def _select_primary_strategy(scores: Dict[str, float]) -> Optional[str]:
        if not scores:
            return None
        return max(scores.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _filter_context(context: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if not isinstance(context, Mapping):
            return {}
        result: Dict[str, Any] = {}
        for key, value in context.items():
            if isinstance(value, (str, int, float, bool)):
                result[str(key)] = value
        return result

    @staticmethod
    def _normalise_emotion(emotion: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if not isinstance(emotion, Mapping):
            return {}
        summary: Dict[str, Any] = {}
        primary = emotion.get("primary")
        if primary is not None:
            summary["primary"] = str(primary)
        for key in ("intensity", "mood", "valence", "arousal"):
            if key in emotion:
                val = _safe_float(emotion[key])
                if val is not None:
                    summary[key] = val
        if "confidence" in emotion:
            val = _safe_float(emotion["confidence"])
            if val is not None:
                summary["confidence"] = val
        return summary


__all__ = ["SelfModel", "SelfModelState", "StrategyStats", "CapabilityStats"]
