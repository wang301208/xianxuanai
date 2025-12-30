from __future__ import annotations

import json
import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from typing import Deque

from .state import (
    CuriosityState,
    EmotionSnapshot,
    PerceptionSnapshot,
    PersonalityProfile,
)

logger = logging.getLogger(__name__)


def _inject_knowledge_metadata(
    context: Optional[Dict[str, Any]], metadata: Dict[str, Any], tags: List[str]
) -> None:
    """Augment metadata and tags with knowledge retrieval context if available."""

    if not isinstance(context, dict):
        return
    knowledge_context = context.get("knowledge_context")
    if knowledge_context:
        metadata["knowledge_context"] = knowledge_context
        knowledge_query = context.get("knowledge_query")
        if isinstance(knowledge_query, str) and knowledge_query.strip():
            metadata["knowledge_query"] = knowledge_query.strip()
        if "knowledge-informed" not in tags:
            tags.append("knowledge-informed")

    def _truncate(value: str, *, limit: int = 256) -> str:
        value = (value or "").strip()
        if len(value) <= limit:
            return value
        return value[: max(0, limit - 1)] + "…"

    def _prune_records(records: Any, *, limit: int = 3) -> List[Dict[str, Any]]:
        if not isinstance(records, Sequence):
            return []
        pruned: List[Dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            payload: Dict[str, Any] = {}
            if "id" in record:
                payload["id"] = record["id"]
            text = record.get("text")
            if isinstance(text, str) and text.strip():
                payload["text"] = _truncate(text, limit=240)
            score = record.get("score")
            try:
                if score is not None:
                    payload["score"] = float(score)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                pass
            metadata_payload = record.get("metadata")
            if isinstance(metadata_payload, dict) and metadata_payload:
                limited_meta: Dict[str, Any] = {}
                for index, (key, value) in enumerate(metadata_payload.items()):
                    if index >= 3:
                        break
                    limited_meta[key] = value
                if limited_meta:
                    payload["metadata"] = limited_meta
            if payload:
                pruned.append(payload)
            if len(pruned) >= limit:
                break
        return pruned

    def _prune_known_facts(facts: Any, *, limit: int = 5) -> List[Dict[str, Any]]:
        if not isinstance(facts, Sequence):
            return []
        pruned: List[Dict[str, Any]] = []
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            payload: Dict[str, Any] = {}
            for key in ("subject", "predicate", "object"):
                value = fact.get(key)
                if isinstance(value, str) and value.strip():
                    payload[key] = _truncate(value, limit=160)
            if payload:
                pruned.append(payload)
            if len(pruned) >= limit:
                break
        return pruned

    memory_retrieval = context.get("memory_retrieval")
    memory_query = context.get("memory_query")
    memory_records = context.get("memory_records")
    memory_known_facts = context.get("memory_known_facts")
    if isinstance(memory_retrieval, dict):
        memory_query = memory_query or memory_retrieval.get("query")
        memory_records = memory_records or memory_retrieval.get("records")
        memory_known_facts = memory_known_facts or memory_retrieval.get("known_facts")

    memory_payload: Dict[str, Any] = {}
    if isinstance(memory_query, str) and memory_query.strip():
        memory_payload["query"] = _truncate(memory_query)
    pruned_records = _prune_records(memory_records)
    if pruned_records:
        memory_payload["records"] = pruned_records
    pruned_known = _prune_known_facts(memory_known_facts)
    if pruned_known:
        memory_payload["known_facts"] = pruned_known

    if memory_payload:
        metadata["memory_retrieval"] = memory_payload
        if "memory-informed" not in tags:
            tags.append("memory-informed")

    def _prune_causal_relations(relations: Any, *, limit: int = 4) -> List[Dict[str, Any]]:
        if not isinstance(relations, Sequence):
            return []
        pruned: List[Dict[str, Any]] = []
        for relation in relations:
            cause: str = ""
            effect: str = ""
            weight: Any = None
            description: Optional[str] = None
            metadata_payload: Any = None
            if hasattr(relation, "cause") or hasattr(relation, "effect"):
                cause = str(getattr(relation, "cause", "") or "").strip()
                effect = str(getattr(relation, "effect", "") or "").strip()
                weight = getattr(relation, "weight", None)
                metadata_payload = getattr(relation, "metadata", None)
            elif isinstance(relation, dict):
                cause = str(relation.get("cause") or relation.get("source") or "").strip()
                effect = str(relation.get("effect") or relation.get("target") or "").strip()
                weight = relation.get("weight")
                metadata_payload = relation.get("metadata")
                description = relation.get("description") or relation.get("summary")
            elif isinstance(relation, (tuple, list)) and relation:
                cause = str(relation[0] if len(relation) > 0 else "").strip()
                effect = str(relation[1] if len(relation) > 1 else "").strip()
                weight = relation[2] if len(relation) > 2 else None
            payload: Dict[str, Any] = {}
            if cause:
                payload["cause"] = _truncate(cause, limit=160)
            if effect:
                payload["effect"] = _truncate(effect, limit=160)
            try:
                if weight is not None:
                    payload["weight"] = float(weight)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                payload["weight"] = weight
            if description:
                payload["description"] = _truncate(str(description), limit=220)
            if isinstance(metadata_payload, dict) and metadata_payload:
                limited_meta: Dict[str, Any] = {}
                for index, (key, value) in enumerate(metadata_payload.items()):
                    if index >= 3:
                        break
                    limited_meta[key] = value
                if limited_meta:
                    payload["metadata"] = limited_meta
            if payload:
                pruned.append(payload)
            if len(pruned) >= limit:
                break
        return pruned

    def _prune_causal_paths(
        paths: Any,
        *,
        limit: int = 2,
        step_limit: int = 6,
    ) -> List[Dict[str, Any]]:
        if not isinstance(paths, Sequence):
            return []
        pruned: List[Dict[str, Any]] = []
        for entry in paths:
            cause: str = ""
            effect: str = ""
            sequence: Sequence[Any] | None = None
            if isinstance(entry, dict):
                cause = str(entry.get("cause") or "").strip()
                effect = str(entry.get("effect") or "").strip()
                raw_path = entry.get("path") or entry.get("steps")
                if isinstance(raw_path, Sequence) and not isinstance(raw_path, (str, bytes)):
                    sequence = raw_path
            elif isinstance(entry, (tuple, list)):
                if entry:
                    cause = str(entry[0]).strip()
                if len(entry) > 1:
                    effect = str(entry[1]).strip()
                if len(entry) > 2 and isinstance(entry[2], Sequence):
                    sequence = entry[2]
            if sequence is None and isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
                sequence = entry
            payload: Dict[str, Any] = {}
            if cause:
                payload["cause"] = _truncate(cause, limit=160)
            if effect:
                payload["effect"] = _truncate(effect, limit=160)
            if sequence:
                collapsed: List[str] = []
                for node in sequence:
                    if len(collapsed) >= step_limit:
                        break
                    collapsed.append(_truncate(str(node), limit=160))
                if collapsed:
                    payload["path"] = collapsed
            if payload:
                pruned.append(payload)
            if len(pruned) >= limit:
                break
        return pruned

    causal_relations = context.get("causal_relations")
    if not causal_relations and isinstance(knowledge_context, dict):
        causal_relations = knowledge_context.get("causal_relations")
    pruned_causal = _prune_causal_relations(causal_relations)
    if pruned_causal:
        metadata["causal_relations"] = pruned_causal
        if "causal-informed" not in tags:
            tags.append("causal-informed")

    causal_paths = context.get("causal_paths")
    if not causal_paths and isinstance(knowledge_context, dict):
        causal_paths = knowledge_context.get("causal_paths")
    pruned_paths = _prune_causal_paths(causal_paths)
    if pruned_paths:
        metadata["causal_paths"] = pruned_paths
        if "causal-informed" not in tags:
            tags.append("causal-informed")

    causal_focus = context.get("causal_focus")
    if isinstance(causal_focus, str) and causal_focus.strip():
        metadata.setdefault("causal_focus", _truncate(causal_focus.strip(), limit=160))

def default_plan_for_intention(
    intention: str,
    focus: Optional[str],
    context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Generate a lightweight plan for a given intention."""

    context = context or {}
    threat = float(context.get("threat", 0.0) or 0.0)
    safety = float(context.get("safety", 0.0) or 0.0)
    novelty = float(context.get("novelty", 0.0) or 0.0)
    social = float(context.get("social", 0.0) or 0.0)
    plan: List[str] = []
    if intention == "explore":
        plan = [
            "scan_environment",
            f"focus_{focus}" if focus else "sample_new_modalities",
            "log_novelty",
        ]
        if novelty < 0.3:
            plan.append("expand_search_radius")
        else:
            plan.append("synthesise_discovery_brief")
    elif intention == "approach":
        plan = [
            "identify_positive_stimulus",
            f"move_towards_{focus}" if focus else "establish_focus",
            "engage",
        ]
        if social > 0.4:
            plan.append("synchronise_with_allies")
        if safety < 0.3:
            plan.append("verify_safety_corridor")
    elif intention == "withdraw":
        plan = ["assess_risk", "increase_distance", "seek_support"]
        if threat > 0.6:
            plan.insert(0, "raise_alert")
        if safety < 0.2:
            plan.append("establish_emergency_contact")
    else:
        plan = ["monitor_sensory_streams", "maintain_attention"]
        if threat > 0.4:
            plan.append("prepare_contingency")
        if novelty > 0.5:
            plan.append("capture_observations")

    task = context.get("task")
    if task:
        plan.append(f"respect_task_{task}")
    if safety > 0.7 and "log_positive_feedback" not in plan:
        plan.append("log_positive_feedback")
    return [step for step in plan if step]


@dataclass
class CognitiveDecision:
    """Container for policy-driven cognitive decisions."""

    intention: str
    confidence: float
    plan: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    focus: Optional[str] = None
    summary: str = ""
    thought_trace: List[str] = field(default_factory=list)
    perception_summary: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CognitivePolicy:
    """Interface for pluggable cognitive intention selection policies."""

    def select_intention(
        self,
        perception: PerceptionSnapshot,
        summary: Dict[str, float],
        emotion: EmotionSnapshot,
        personality: PersonalityProfile,
        curiosity: CuriosityState,
        context: Dict[str, Any],
        learning_prediction: Optional[Dict[str, float]] = None,
        history: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> CognitiveDecision:
        raise NotImplementedError


class HeuristicCognitivePolicy(CognitivePolicy):
    """Default heuristic policy mirroring the legacy weighting logic."""

    def _build_tags(
        self,
        intention: str,
        confidence: float,
        curiosity: CuriosityState,
        focus: Optional[str],
    ) -> List[str]:
        tags = [intention]
        if confidence >= 0.65:
            tags.append("high-confidence")
        if curiosity.last_novelty > 0.6:
            tags.append("novelty-driven")
        if focus:
            tags.append(f"focus-{focus}")
        return tags

    def select_intention(
        self,
        perception: PerceptionSnapshot,
        summary: Dict[str, float],
        emotion: EmotionSnapshot,
        personality: PersonalityProfile,
        curiosity: CuriosityState,
        context: Dict[str, Any],
        learning_prediction: Optional[Dict[str, float]] = None,
        history: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> CognitiveDecision:
        focus = max(summary, key=summary.get) if summary else None
        options = {
            "observe": 0.2 + (1 - abs(emotion.dimensions.get("valence", 0.0))) * 0.3,
            "approach": 0.2 + emotion.intent_bias.get("approach", 0.0),
            "withdraw": 0.2 + emotion.intent_bias.get("withdraw", 0.0),
            "explore": 0.2
            + emotion.intent_bias.get("explore", 0.0)
            + curiosity.drive * 0.5,
        }
        if learning_prediction:
            predicted_load = float(learning_prediction.get("cpu", 0.0))
            resource_pressure = float(learning_prediction.get("memory", 0.0))
            options["observe"] += max(0.0, predicted_load - 0.5) * 0.3
            options["withdraw"] += max(0.0, resource_pressure - 0.5) * 0.2
            options["approach"] += max(0.0, 0.5 - predicted_load) * 0.2
        if context.get("threat", 0.0) > 0.4:
            options["withdraw"] += 0.3
        if context.get("safety", 0.0) > 0.5:
            options["approach"] += 0.2
        options["explore"] *= 0.5 + personality.modulation_weight("explore")
        options["approach"] *= 0.5 + personality.modulation_weight("social")
        options["withdraw"] *= 0.5 + personality.modulation_weight("caution")
        options["observe"] *= 0.5 + personality.modulation_weight("persist")
        total = sum(options.values()) or 1.0
        weights = {key: value / total for key, value in options.items()}
        intention = max(weights.items(), key=lambda item: item[1])[0]
        confidence = weights[intention]
        plan = default_plan_for_intention(intention, focus, context)
        tags = self._build_tags(intention, confidence, curiosity, focus)
        thought_trace = [
            f"focus={focus or 'none'}",
            f"intention={intention}",
            f"emotion={emotion.primary.value}:{emotion.intensity:.2f}",
            f"curiosity={curiosity.drive:.2f}",
        ]
        if learning_prediction:
            thought_trace.append(
                f"predicted_cpu={float(learning_prediction.get('cpu', 0.0)):.2f}"
            )
            thought_trace.append(
                f"predicted_mem={float(learning_prediction.get('memory', 0.0)):.2f}"
            )
        summary_text = (
            ", ".join(f"{k}:{v:.2f}" for k, v in summary.items())
            or "no-salient-modalities"
        )
        metadata = {"policy": "heuristic"}
        _inject_knowledge_metadata(context, metadata, tags)
        return CognitiveDecision(
            intention=intention,
            confidence=confidence,
            plan=plan,
            weights=weights,
            tags=tags,
            focus=focus,
            summary=summary_text,
            thought_trace=thought_trace,
            perception_summary=dict(summary),
            metadata=metadata,
        )


class StructuredPlanner:
    """Deterministic planner assembling multi-stage cognitive plans."""

    name = "structured"

    def __init__(self, min_steps: int = 3) -> None:
        self.min_steps = max(1, int(min_steps))

    def generate(
        self,
        intention: str,
        focus: Optional[str],
        context: Dict[str, Any],
        perception: PerceptionSnapshot,
        summary: Dict[str, float],
        emotion: EmotionSnapshot,
        curiosity: CuriosityState,
        history: Optional[Sequence[Dict[str, Any]]] = None,
        learning_prediction: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        plan: List[str] = []
        focus_token = focus or (max(summary, key=summary.get) if summary else None)
        intention_key = (intention or "observe").lower()
        novelty = context.get("novelty", curiosity.last_novelty)
        threat = context.get("threat", 0.0)
        safety = context.get("safety", 0.0)
        social = context.get("social", 0.0)

        def push(step: str) -> None:
            if step:
                plan.append(step)

        memory_records = context.get("memory_records")
        memory_known_facts = context.get("memory_known_facts")
        if memory_records:
            push("consult_retrieved_memory")
        if memory_known_facts:
            push("integrate_known_facts")

        causal_relations = context.get("causal_relations")
        causal_paths = context.get("causal_paths")
        if causal_relations:
            push("evaluate_causal_relations")
            push("simulate_causal_outcomes")
        if causal_paths:
            push("trace_causal_chain")

        if intention_key == "observe":
            push("stabilise_attention")
            push("collect_multimodal_snapshot")
            if focus_token:
                push(f"inspect_{focus_token}_salience")
            else:
                push("scan_salient_modalities")
            push("update_world_model")
            if learning_prediction and float(learning_prediction.get("cpu", 0.0)) > 0.6:
                push("shed_low_priority_tasks")
        elif intention_key == "approach":
            push("identify_positive_target")
            if focus_token:
                push(f"focus_{focus_token}")
            else:
                push("select_focus_modality")
            push("compute_safe_path")
            if social > 0.4:
                push("establish_social_contact")
            push("engage_target")
        elif intention_key == "withdraw":
            push("elevate_alert_state")
            push("assess_risk_vectors")
            push("select_evasive_route")
            if focus_token:
                push(f"avoid_{focus_token}")
            else:
                push("choose_safe_direction")
            push("seek_support_channel")
        elif intention_key == "explore":
            if focus_token:
                push(f"probe_{focus_token}")
            else:
                push("discover_salient_focus")
            push("scan_unexplored_modalities")
            push("sample_novel_patterns")
            push("log_novelty_metrics")
            if curiosity.drive > 0.6:
                push("expand_search_radius")
        else:
            for step in default_plan_for_intention(intention_key, focus_token, context):
                push(step)

        if history:
            recent = [str(item.get("intention", "")).lower() for item in history if item]
            if recent.count(intention_key) >= 3:
                push("refresh_strategy_model")
        if threat > 0.6 and intention_key != "withdraw":
            push("publish_alert_status")
        if novelty > 0.6 and intention_key == "explore":
            push("synthesise_novelty_brief")
        if safety > 0.7 and intention_key == "approach":
            push("capture_positive_feedback")

        deduped: List[str] = []
        seen: set[str] = set()
        for step in plan:
            if step and step not in seen:
                deduped.append(step)
                seen.add(step)
        while len(deduped) < self.min_steps:
            filler = "log_cognitive_trace" if "log_cognitive_trace" not in seen else "archive_cognitive_trace"
            deduped.append(filler)
            seen.add(filler)
        if "archive_cognitive_trace" not in seen:
            deduped.append("archive_cognitive_trace")
        return deduped


@dataclass
class _Transition:
    """Experience tuple stored for meta-learning and replay."""

    state: Tuple[Any, ...]
    action: str
    reward: float
    next_state: Tuple[Any, ...]
    terminal: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReinforcementCognitivePolicy(CognitivePolicy):
    """On-line reinforcement learning policy using tabular Q-learning."""

    INTENTIONS: Tuple[str, ...] = ("observe", "approach", "withdraw", "explore")

    def __init__(
        self,
        *,
        learning_rate: float = 0.08,
        discount: float = 0.85,
        exploration: float = 0.12,
        exploration_decay: float = 0.995,
        min_exploration: float = 0.02,
        reward_beta: float = 0.85,
        exploration_smoothing: float = 0.35,
        planner: Optional[StructuredPlanner] = None,
        fallback: Optional[CognitivePolicy] = None,
        replay_buffer_size: int = 256,
        replay_batch_size: int = 16,
        replay_iterations: int = 1,
    ) -> None:
        self.learning_rate = max(1e-3, float(learning_rate))
        self.discount = max(0.0, min(0.99, float(discount)))
        self.exploration = max(0.0, min(1.0, float(exploration)))
        self.exploration_decay = max(0.9, min(0.9999, float(exploration_decay)))
        self.min_exploration = max(0.0, min(0.2, float(min_exploration)))
        self.reward_beta = max(0.0, min(0.999, float(reward_beta)))
        self._exploration_smoothing = max(0.0, min(0.99, float(exploration_smoothing)))
        self.q_table: Dict[Tuple[Any, ...], Dict[str, float]] = {}
        self.last_state: Optional[Tuple[Any, ...]] = None
        self.last_action: Optional[str] = None
        self.last_value: float = 0.0
        self.planner = planner or StructuredPlanner(min_steps=4)
        self.fallback = fallback or ProductionCognitivePolicy(planner=self.planner)
        self._temperature = 1.0
        self._reward_ema: Optional[float] = None
        self._adaptive_exploration = self.exploration
        self._temperature_bounds = (0.2, 1.5)
        self._experience_buffer: Deque[_Transition] = deque(
            maxlen=max(1, int(replay_buffer_size))
        )
        self._replay_batch_size = max(1, int(replay_batch_size))
        self._replay_iterations = max(1, int(replay_iterations))

    # ------------------------------------------------------------------
    def _store_transition(
        self,
        state: Tuple[Any, ...],
        action: str,
        reward: float,
        next_state: Tuple[Any, ...],
        *,
        terminal: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist an experience tuple for replay-based refinement."""

        transition = _Transition(
            state=state,
            action=action,
            reward=float(reward),
            next_state=next_state,
            terminal=bool(terminal),
            metadata=dict(metadata or {}),
        )
        self._experience_buffer.append(transition)

    def _bucket(
        self,
        value: float,
        bins: int = 5,
        *,
        span: Tuple[float, float] = (-1.0, 1.0),
    ) -> int:
        low, high = span
        value = max(low, min(high, float(value)))
        step = (high - low) / max(1, bins)
        if step <= 0:
            return 0
        return int(math.floor((value - low) / step))

    def _encode_state(
        self,
        summary: Dict[str, float],
        emotion: EmotionSnapshot,
        personality: PersonalityProfile,
        curiosity: CuriosityState,
        context: Dict[str, Any],
        learning_prediction: Optional[Dict[str, float]],
    ) -> Tuple[Any, ...]:
        focus = max(summary, key=summary.get) if summary else "none"
        valence = emotion.dimensions.get("valence", 0.0)
        arousal = emotion.dimensions.get("arousal", 0.5)
        dominance = emotion.dimensions.get("dominance", 0.0)
        curiosity_drive = curiosity.drive
        novelty = curiosity.last_novelty
        threat = float(context.get("threat", 0.0) or 0.0)
        safety = float(context.get("safety", 0.0) or 0.0)
        social = float(context.get("social", 0.0) or 0.0)
        cpu = float((learning_prediction or {}).get("cpu", 0.0))
        memory = float((learning_prediction or {}).get("memory", 0.0))
        personality_bins = tuple(
            self._bucket(personality.traits.get(name, 0.0), 5)
            for name in sorted(personality.traits)
        )
        state = (
            focus,
            self._bucket(valence, 7),
            self._bucket(arousal, 7, span=(0.0, 1.0)),
            self._bucket(dominance, 7),
            self._bucket(curiosity_drive, 7, span=(0.0, 1.0)),
            self._bucket(novelty, 7, span=(0.0, 1.0)),
            self._bucket(threat, 7, span=(0.0, 1.0)),
            self._bucket(safety, 7, span=(0.0, 1.0)),
            self._bucket(social, 7, span=(0.0, 1.0)),
            self._bucket(cpu, 7, span=(0.0, 1.0)),
            self._bucket(memory, 7, span=(0.0, 1.0)),
            personality_bins,
        )
        return state

    def _ensure_state(self, state: Tuple[Any, ...]) -> Dict[str, float]:
        if state not in self.q_table:
            self.q_table[state] = {intent: 0.0 for intent in self.INTENTIONS}
        return self.q_table[state]

    def _policy_distribution(self, q_values: Dict[str, float]) -> Dict[str, float]:
        if not q_values:
            return {intent: 1.0 / len(self.INTENTIONS) for intent in self.INTENTIONS}
        scaled = {
            intent: value / max(1e-6, self._temperature)
            for intent, value in q_values.items()
        }
        max_value = max(scaled.values())
        exp_values = {intent: math.exp(value - max_value) for intent, value in scaled.items()}
        total = sum(exp_values.values()) or 1.0
        return {intent: exp_value / total for intent, exp_value in exp_values.items()}

    def _derive_reward(
        self,
        emotion: EmotionSnapshot,
        curiosity: CuriosityState,
        context: Dict[str, Any],
    ) -> float:
        reward = 0.0
        reward += float(emotion.dimensions.get("valence", 0.0)) * 0.6
        reward += float(emotion.dimensions.get("dominance", 0.0)) * 0.2
        reward += float(curiosity.drive) * 0.1
        reward += float(curiosity.last_novelty) * 0.1
        reward -= float(context.get("threat", 0.0) or 0.0) * 0.4
        reward += float(context.get("safety", 0.0) or 0.0) * 0.3
        if "reward" in context:
            try:
                reward = float(context["reward"])
            except (TypeError, ValueError):
                pass
        elif "feedback" in context:
            try:
                reward += float(context["feedback"])
            except (TypeError, ValueError):
                pass
        return max(-1.0, min(1.0, reward))

    def _update_q_value(
        self,
        state: Tuple[Any, ...],
        action: str,
        reward: float,
        next_state: Tuple[Any, ...],
    ) -> None:
        table = self._ensure_state(state)
        current = table.get(action, 0.0)
        next_values = self._ensure_state(next_state)
        best_next = max(next_values.values()) if next_values else 0.0
        target = reward + self.discount * best_next
        updated = current + self.learning_rate * (target - current)
        table[action] = updated

    def _apply_learning(
        self,
        reward: float,
        next_state: Tuple[Any, ...],
    ) -> None:
        if self.last_state is None or self.last_action is None:
            return
        try:
            self._update_q_value(self.last_state, self.last_action, reward, next_state)
            self._store_transition(
                self.last_state,
                self.last_action,
                reward,
                next_state,
                metadata={"source": "intrinsic"},
            )
        finally:
            self.last_value = reward

    # ------------------------------------------------------------------
    def integrate_external_feedback(
        self,
        reward: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Blend environment feedback into the learned value estimates."""

        if self.last_state is None or self.last_action is None:
            return
        bounded_reward = max(-1.0, min(1.0, float(reward)))
        self._update_q_value(self.last_state, self.last_action, bounded_reward, self.last_state)
        meta = dict(metadata or {})
        meta.setdefault("source", "extrinsic")
        meta.setdefault("success", bool(success))
        self._store_transition(
            self.last_state,
            self.last_action,
            bounded_reward,
            self.last_state,
            terminal=True,
            metadata=meta,
        )
        if success:
            self._adaptive_exploration = max(
                self.min_exploration,
                self._adaptive_exploration * 0.9,
            )
        else:
            self._adaptive_exploration = min(1.0, self._adaptive_exploration * 1.1 + 0.05)

    # ------------------------------------------------------------------
    def replay_from_buffer(
        self,
        *,
        batch_size: Optional[int] = None,
        iterations: Optional[int] = None,
    ) -> int:
        """Perform batched Q-updates from the accumulated experience buffer."""

        if not self._experience_buffer:
            return 0
        batch = max(1, int(batch_size or self._replay_batch_size))
        rounds = max(1, int(iterations or self._replay_iterations))
        updates = 0
        for _ in range(rounds):
            if not self._experience_buffer:
                break
            samples = random.sample(
                list(self._experience_buffer),
                k=min(batch, len(self._experience_buffer)),
            )
            for transition in samples:
                self._update_q_value(
                    transition.state,
                    transition.action,
                    transition.reward,
                    transition.next_state,
                )
                updates += 1
        return updates

    def select_intention(
        self,
        perception: PerceptionSnapshot,
        summary: Dict[str, float],
        emotion: EmotionSnapshot,
        personality: PersonalityProfile,
        curiosity: CuriosityState,
        context: Dict[str, Any],
        learning_prediction: Optional[Dict[str, float]] = None,
        history: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> CognitiveDecision:
        context = dict(context or {})
        current_state = self._encode_state(
            summary, emotion, personality, curiosity, context, learning_prediction
        )
        reward = self._derive_reward(emotion, curiosity, context)
        prev_ema = self._reward_ema
        if prev_ema is None:
            reward_delta = 0.0
            self._reward_ema = reward
        else:
            reward_delta = reward - prev_ema
            self._reward_ema = self.reward_beta * prev_ema + (1 - self.reward_beta) * reward
        self._apply_learning(reward, current_state)
        q_values = self._ensure_state(current_state)

        base_temperature = (
            1.0
            - float(emotion.dimensions.get("valence", 0.0)) * 0.3
            + float(curiosity.last_novelty) * 0.2
            + float(context.get("threat", 0.0) or 0.0) * 0.3
        )
        delta_term = -0.25 * max(-1.0, min(1.0, reward_delta))
        lower, upper = self._temperature_bounds
        self._temperature = max(lower, min(upper, base_temperature + delta_term))

        base_exploration = self.exploration
        if history:
            base_exploration = max(
                self.min_exploration,
                self.exploration * (self.exploration_decay ** len(history)),
            )
        adaptive_target = base_exploration
        if prev_ema is not None:
            if reward_delta > 0.01:
                adaptive_target *= max(0.1, 1.0 - min(0.5, reward_delta * 0.5))
            elif reward_delta < -0.01:
                adaptive_target *= 1.0 + min(0.6, abs(reward_delta) * 0.7)
        smoothing = self._exploration_smoothing
        self._adaptive_exploration = max(
            self.min_exploration,
            min(
                0.95,
                smoothing * self._adaptive_exploration
                + (1.0 - smoothing) * adaptive_target,
            ),
        )
        exploration_rate = self._adaptive_exploration
        if random.random() < exploration_rate:
            intention = random.choice(self.INTENTIONS)
        else:
            intention = max(q_values.items(), key=lambda item: item[1])[0]
        distribution = self._policy_distribution(q_values)
        confidence = distribution.get(intention, 0.25)
        focus = max(summary, key=summary.get) if summary else None
        plan = self.planner.generate(
            intention,
            focus,
            context,
            perception,
            summary,
            emotion,
            curiosity,
            history=history,
            learning_prediction=learning_prediction,
        )
        tags = [intention, "reinforcement-policy"]
        if focus:
            tags.append(f"focus-{focus}")
        if exploration_rate > self.min_exploration * 1.1:
            tags.append("exploring")
        thought_trace = [
            f"reward={reward:.2f}",
            f"exploration={exploration_rate:.2f}",
            f"valence={emotion.dimensions.get('valence', 0.0):.2f}",
            f"arousal={emotion.dimensions.get('arousal', 0.0):.2f}",
            f"novelty={curiosity.last_novelty:.2f}",
        ]
        if learning_prediction:
            thought_trace.append(
                f"cpu={float(learning_prediction.get('cpu', 0.0)):.2f}"
            )
            thought_trace.append(
                f"mem={float(learning_prediction.get('memory', 0.0)):.2f}"
            )

        summary_text = ", ".join(f"{k}:{v:.2f}" for k, v in summary.items()) or "no-salient-modalities"
        metadata = {
            "policy": "reinforcement",
            "reward": reward,
            "reward_ema": self._reward_ema,
            "reward_delta": reward_delta,
            "exploration_rate": exploration_rate,
            "adaptive_exploration": exploration_rate,
        }
        _inject_knowledge_metadata(context, metadata, tags)
        decision = CognitiveDecision(
            intention=intention,
            confidence=confidence,
            plan=plan,
            weights=distribution,
            tags=tags,
            focus=focus,
            summary=summary_text,
            thought_trace=thought_trace,
            perception_summary=dict(summary),
            metadata=metadata,
        )
        self.last_state = current_state
        self.last_action = intention
        self.exploration = max(self.min_exploration, self.exploration * self.exploration_decay)
        return decision


class BanditCognitivePolicy(CognitivePolicy):
    """Contextual bandit for intention selection with online updates.

    When deep RL stacks (e.g. PPO/A3C via PyTorch) are unavailable, this policy
    can still be trained online from (state, intention, reward) tuples captured
    during runtime and updated in a background thread.
    """

    INTENTIONS: Tuple[str, ...] = ("observe", "approach", "withdraw", "explore")

    def __init__(
        self,
        *,
        exploration: float = 0.08,
        min_exploration: float = 0.02,
        exploration_decay: float = 0.997,
        planner: Optional[StructuredPlanner] = None,
        fallback: Optional[CognitivePolicy] = None,
    ) -> None:
        self.exploration = max(0.0, min(1.0, float(exploration)))
        self.min_exploration = max(0.0, min(0.4, float(min_exploration)))
        self.exploration_decay = max(0.9, min(0.9999, float(exploration_decay)))
        self.planner = planner or StructuredPlanner(min_steps=4)
        self.fallback = fallback or ProductionCognitivePolicy(planner=self.planner)
        self._adaptive_exploration = self.exploration
        self._stats: Dict[str, Dict[str, Dict[str, float]]] = {}

    # ------------------------------------------------------------------
    def _bucket(self, value: float, bins: int = 6, *, span: Tuple[float, float] = (-1.0, 1.0)) -> int:
        low, high = span
        value = max(low, min(high, float(value)))
        step = (high - low) / max(1, bins)
        if step <= 0:
            return 0
        return int(math.floor((value - low) / step))

    def _encode_state(
        self,
        summary: Dict[str, float],
        emotion: EmotionSnapshot,
        curiosity: CuriosityState,
        context: Dict[str, Any],
        learning_prediction: Optional[Dict[str, float]],
    ) -> Tuple[Any, ...]:
        focus = max(summary, key=summary.get) if summary else "none"
        valence = float(emotion.dimensions.get("valence", 0.0))
        arousal = float(emotion.dimensions.get("arousal", 0.5))
        novelty = float(curiosity.last_novelty)
        threat = float(context.get("threat", 0.0) or 0.0)
        safety = float(context.get("safety", 0.0) or 0.0)
        cpu = float((learning_prediction or {}).get("cpu", 0.0))
        memory = float((learning_prediction or {}).get("memory", 0.0))
        return (
            focus,
            self._bucket(valence, span=(-1.0, 1.0)),
            self._bucket(arousal, span=(0.0, 1.0)),
            self._bucket(novelty, span=(0.0, 1.0)),
            self._bucket(threat, span=(0.0, 1.0)),
            self._bucket(safety, span=(0.0, 1.0)),
            self._bucket(cpu, span=(0.0, 1.0)),
            self._bucket(memory, span=(0.0, 1.0)),
        )

    @staticmethod
    def _state_key(state: Tuple[Any, ...]) -> str:
        return "|".join(str(item) for item in state)

    def _expected_reward(self, state_key: str, intention: str) -> float:
        slot = self._stats.get(state_key, {}).get(intention)
        if not slot:
            return 0.0
        count = float(slot.get("n", 0.0))
        if count <= 0:
            return 0.0
        return float(slot.get("reward_sum", 0.0)) / count

    # ------------------------------------------------------------------
    def update_from_trajectories(self, trajectories: Sequence[Dict[str, Any]]) -> int:
        """Update the bandit statistics from serialized trajectory payloads."""

        updated = 0
        for traj in trajectories:
            if not isinstance(traj, dict):
                continue
            features = traj.get("features")
            if not isinstance(features, dict):
                continue
            state_key = features.get("state_key")
            if not isinstance(state_key, str) or not state_key.strip():
                continue
            intention = traj.get("intention")
            if intention not in self.INTENTIONS:
                continue
            reward = traj.get("reward")
            if reward is None:
                reward = 1.0 if bool(traj.get("success", False)) else 0.0
            try:
                reward_value = float(reward)
            except (TypeError, ValueError):
                continue

            action_stats = self._stats.setdefault(state_key, {}).setdefault(
                intention, {"n": 0.0, "reward_sum": 0.0}
            )
            action_stats["n"] = float(action_stats.get("n", 0.0)) + 1.0
            action_stats["reward_sum"] = float(action_stats.get("reward_sum", 0.0)) + reward_value
            updated += 1

        if updated:
            self._adaptive_exploration = max(
                self.min_exploration, self._adaptive_exploration * self.exploration_decay
            )
        return updated

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        payload = {
            "version": 1,
            "intentions": list(self.INTENTIONS),
            "exploration": float(self.exploration),
            "min_exploration": float(self.min_exploration),
            "exploration_decay": float(self.exploration_decay),
            "adaptive_exploration": float(self._adaptive_exploration),
            "stats": self._stats,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        planner: Optional[StructuredPlanner] = None,
        fallback: Optional[CognitivePolicy] = None,
    ) -> "BanditCognitivePolicy":
        target = Path(path)
        policy = cls(planner=planner, fallback=fallback)
        if not target.exists():
            return policy
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            return policy
        if not isinstance(payload, dict):
            return policy
        try:
            policy.exploration = float(payload.get("exploration", policy.exploration))
            policy.min_exploration = float(payload.get("min_exploration", policy.min_exploration))
            policy.exploration_decay = float(payload.get("exploration_decay", policy.exploration_decay))
            policy._adaptive_exploration = float(payload.get("adaptive_exploration", policy._adaptive_exploration))
        except (TypeError, ValueError):
            pass
        stats = payload.get("stats")
        if isinstance(stats, dict):
            policy._stats = stats  # type: ignore[assignment]
        return policy

    # ------------------------------------------------------------------
    def select_intention(
        self,
        perception: PerceptionSnapshot,
        summary: Dict[str, float],
        emotion: EmotionSnapshot,
        personality: PersonalityProfile,
        curiosity: CuriosityState,
        context: Dict[str, Any],
        learning_prediction: Optional[Dict[str, float]] = None,
        history: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> CognitiveDecision:
        context = dict(context or {})
        state = self._encode_state(summary, emotion, curiosity, context, learning_prediction)
        state_key = self._state_key(state)

        epsilon = self._adaptive_exploration
        if history:
            epsilon = max(self.min_exploration, epsilon * (self.exploration_decay ** len(history)))

        if state_key not in self._stats:
            return self.fallback.select_intention(
                perception,
                summary,
                emotion,
                personality,
                curiosity,
                context,
                learning_prediction,
                history=history,
            )

        if random.random() < epsilon:
            intention = random.choice(self.INTENTIONS)
        else:
            scored = [(intent, self._expected_reward(state_key, intent)) for intent in self.INTENTIONS]
            scored.sort(key=lambda item: item[1], reverse=True)
            intention = scored[0][0] if scored else "observe"

        confidence = 0.35 + min(0.45, abs(self._expected_reward(state_key, intention)) * 0.3)
        focus = max(summary, key=summary.get) if summary else None
        plan = self.planner.generate(
            intention,
            focus,
            context,
            perception,
            summary,
            emotion,
            curiosity,
            history=history,
            learning_prediction=learning_prediction,
        )
        weights = {k: 0.0 for k in self.INTENTIONS}
        weights[intention] = 1.0
        tags = [intention, "bandit-policy"]
        metadata: Dict[str, Any] = {"policy": "bandit", "state_key": state_key, "epsilon": epsilon}
        _inject_knowledge_metadata(context, metadata, tags)

        return CognitiveDecision(
            intention=intention,
            confidence=float(max(0.0, min(1.0, confidence))),
            plan=plan,
            weights=weights,
            tags=tags,
            focus=focus,
            summary=", ".join(f"{k}:{v:.2f}" for k, v in summary.items()) or "no-salient-modalities",
            thought_trace=[f"policy=bandit", f"state={state_key}", f"epsilon={epsilon:.3f}"],
            perception_summary=dict(summary),
            metadata=metadata,
        )


class ProductionCognitivePolicy(CognitivePolicy):
    """Linear policy trained on synthetic roll-outs for deployment."""

    INTENTIONS: Tuple[str, ...] = ("observe", "approach", "withdraw", "explore")

    def __init__(
        self,
        weight_matrix: Optional[Sequence[Sequence[float]]] = None,
        temperature: float = 1.0,
        planner: Optional[StructuredPlanner] = None,
        fallback: Optional[CognitivePolicy] = None,
    ) -> None:
        self.temperature = max(0.1, float(temperature))
        self.weight_matrix: List[List[float]] = [
            list(row)
            for row in (
                weight_matrix if weight_matrix is not None else self._default_weights()
            )
        ]
        self.planner = planner or StructuredPlanner()
        self.fallback = fallback or HeuristicCognitivePolicy()

    def _default_weights(self) -> List[List[float]]:
        return [
            [
                0.40,
                -0.30,
                0.10,
                -0.10,
                -0.05,
                -0.05,
                -0.20,
                0.10,
                -0.15,
                0.05,
                -0.50,
                0.20,
                -0.10,
                -0.05,
                0.10,
                0.25,
                -0.10,
                0.35,
                -0.05,
                0.05,
                0.05,
                0.05,
                0.04,
                0.02,
                0.15,
                -0.05,
                0.10,
                -0.10,
                0.05,
                0.20,
                0.80,
                0.60,
                -0.20,
                0.10,
                -0.10,
                0.20,
                0.10,
                -0.10,
                0.30,
                -0.20,
                -0.40,
                -0.05,
            ],
            [
                -0.10,
                1.40,
                0.30,
                0.20,
                0.50,
                0.30,
                1.20,
                -0.40,
                0.30,
                -0.20,
                -0.60,
                0.80,
                0.20,
                0.90,
                0.30,
                -0.20,
                0.20,
                -0.30,
                0.15,
                0.20,
                0.15,
                0.10,
                0.05,
                0.05,
                0.05,
                0.40,
                0.30,
                0.80,
                0.40,
                -0.40,
                -0.30,
                -0.20,
                0.50,
                -0.20,
                0.20,
                -0.10,
                0.25,
                0.10,
                0.60,
                0.50,
                -0.40,
                0.35,
            ],
            [
                -0.20,
                -1.20,
                0.40,
                -0.50,
                0.30,
                -0.20,
                -0.30,
                1.10,
                -0.20,
                0.10,
                1.40,
                -0.50,
                0.10,
                -0.20,
                -0.10,
                0.20,
                -0.15,
                0.25,
                -0.05,
                0.05,
                0.05,
                0.08,
                0.04,
                0.02,
                0.05,
                -0.20,
                -0.10,
                -0.30,
                -0.05,
                0.80,
                0.20,
                0.15,
                -0.10,
                0.50,
                -0.10,
                -0.05,
                0.10,
                0.05,
                -0.60,
                -0.70,
                0.80,
                0.30,
            ],
            [
                -0.05,
                0.40,
                0.60,
                0.20,
                0.35,
                0.15,
                0.20,
                -0.30,
                1.20,
                -0.10,
                -0.20,
                0.30,
                1.30,
                0.25,
                0.10,
                -0.30,
                1.00,
                -0.40,
                0.80,
                0.15,
                0.12,
                0.10,
                0.08,
                0.20,
                0.08,
                0.90,
                0.20,
                0.30,
                0.10,
                -0.30,
                -0.10,
                -0.15,
                0.10,
                -0.20,
                0.60,
                -0.05,
                0.20,
                0.70,
                0.25,
                0.30,
                -0.20,
                0.18,
            ],
        ]

    def _build_feature_vector(
        self,
        perception: PerceptionSnapshot,
        summary: Dict[str, float],
        emotion: EmotionSnapshot,
        personality: PersonalityProfile,
        curiosity: CuriosityState,
        context: Dict[str, Any],
        learning_prediction: Optional[Dict[str, float]],
        history: Optional[Sequence[Dict[str, Any]]],
    ) -> List[float]:
        valence = emotion.dimensions.get("valence", 0.0)
        arousal = emotion.dimensions.get("arousal", 0.0)
        dominance = emotion.dimensions.get("dominance", 0.0)
        intent_bias = emotion.intent_bias or {}
        context_threat = float(context.get("threat", 0.0))
        context_safety = float(context.get("safety", 0.0))
        context_novelty = float(context.get("novelty", 0.0))
        context_social = float(context.get("social", 0.0))
        context_control = float(context.get("control", 0.0))
        context_fatigue = float(context.get("fatigue", 0.0))
        modalities_count = len(perception.modalities)
        standard_keys = {"vision", "auditory", "somatosensory", "proprioception"}
        other_values = [value for key, value in summary.items() if key not in standard_keys]
        summary_other = sum(other_values) / len(other_values) if other_values else 0.0
        history = history or []
        history_counts = {intent: 0.0 for intent in self.INTENTIONS}
        confidences: List[float] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            intention = str(item.get("intention", "")).lower()
            if intention in history_counts:
                history_counts[intention] += 1.0
            value = item.get("confidence")
            if value is not None:
                try:
                    confidences.append(float(value))
                except (TypeError, ValueError):
                    continue
        total_history = sum(history_counts.values()) or 1.0
        for key in history_counts:
            history_counts[key] /= total_history
        history_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        novelty_delta = curiosity.last_novelty - 0.5
        safety_margin = context_safety - context_threat
        valence_arousal = valence * arousal
        threat_dominance = context_threat * (1.0 - max(0.0, dominance))
        mood_valence = emotion.mood * valence

        return [
            1.0,
            valence,
            arousal,
            dominance,
            emotion.intensity,
            emotion.mood,
            float(intent_bias.get("approach", 0.0)),
            float(intent_bias.get("withdraw", 0.0)),
            float(intent_bias.get("explore", 0.0)),
            float(intent_bias.get("soothe", 0.0)),
            context_threat,
            context_safety,
            context_novelty,
            context_social,
            context_control,
            context_fatigue,
            curiosity.drive,
            curiosity.fatigue,
            curiosity.novelty_preference,
            summary.get("vision", 0.0),
            summary.get("auditory", 0.0),
            summary.get("somatosensory", 0.0),
            summary.get("proprioception", 0.0),
            summary_other,
            min(1.0, modalities_count / 5.0),
            personality.openness,
            personality.conscientiousness,
            personality.extraversion,
            personality.agreeableness,
            personality.neuroticism,
            float(learning_prediction.get("cpu", 0.0)) if learning_prediction else 0.0,
            float(learning_prediction.get("memory", 0.0)) if learning_prediction else 0.0,
            history_counts["approach"],
            history_counts["withdraw"],
            history_counts["explore"],
            history_counts["observe"],
            history_confidence,
            novelty_delta,
            safety_margin,
            valence_arousal,
            threat_dominance,
            mood_valence,
        ]

    def _softmax(self, logits: List[float]) -> List[float]:
        if not logits:
            return []
        scale = self.temperature
        adjusted = [float(logit) / scale for logit in logits]
        max_logit = max(adjusted)
        exps = [math.exp(value - max_logit) for value in adjusted]
        denom = sum(exps) or 1.0
        return [value / denom for value in exps]

    def select_intention(
        self,
        perception: PerceptionSnapshot,
        summary: Dict[str, float],
        emotion: EmotionSnapshot,
        personality: PersonalityProfile,
        curiosity: CuriosityState,
        context: Dict[str, Any],
        learning_prediction: Optional[Dict[str, float]] = None,
        history: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> CognitiveDecision:
        try:
            feature_vector = self._build_feature_vector(
                perception,
                summary,
                emotion,
                personality,
                curiosity,
                context,
                learning_prediction,
                history,
            )
            if any(len(row) != len(feature_vector) for row in self.weight_matrix):
                raise ValueError("weight matrix dimension mismatch")
            logits = [
                sum(weight * feature for weight, feature in zip(row, feature_vector))
                for row in self.weight_matrix
            ]
            probabilities = self._softmax(logits)
            if not probabilities:
                raise ValueError("empty probability distribution")
            index = max(range(len(probabilities)), key=probabilities.__getitem__)
            intention = self.INTENTIONS[index]
            confidence = probabilities[index]
            focus = max(summary, key=summary.get) if summary else None
            plan_steps = self.planner.generate(
                intention,
                focus,
                context,
                perception,
                summary,
                emotion,
                curiosity,
                history,
                learning_prediction,
            )
            weights = {intent: prob for intent, prob in zip(self.INTENTIONS, probabilities)}
            tags = ["policy-production", intention]
            if confidence >= 0.65:
                tags.append("high-confidence")
            if curiosity.drive > 0.6 or context.get("novelty", 0.0) > 0.6:
                tags.append("novelty-driven")
            if focus:
                tags.append(f"focus-{focus}")
            thought_trace = [
                f"features={len(feature_vector)}",
                f"intention={intention}",
                f"confidence={confidence:.2f}",
                f"valence={emotion.dimensions.get('valence', 0.0):.2f}",
                f"novelty={context.get('novelty', curiosity.last_novelty):.2f}",
            ]
            metadata = {
                "policy": "production",
                "policy_version": "1.0",
                "planner": getattr(self.planner, "name", self.planner.__class__.__name__.lower()),
                "temperature": self.temperature,
                "logits": [float(value) for value in logits],
                "probabilities": weights,
            }
            _inject_knowledge_metadata(context, metadata, tags)
            summary_text = ", ".join(f"{k}:{v:.2f}" for k, v in summary.items()) or "no-salient-modalities"
            return CognitiveDecision(
                intention=intention,
                confidence=confidence,
                plan=list(plan_steps),
                weights=weights,
                tags=tags,
                focus=focus,
                summary=summary_text,
                thought_trace=thought_trace,
                perception_summary=dict(summary),
                metadata=metadata,
            )
        except Exception as exc:  # pragma: no cover - defensive safeguard
            logger.debug("Production policy failed: %s", exc)
            decision = self.fallback.select_intention(
                perception,
                summary,
                emotion,
                personality,
                curiosity,
                context,
                learning_prediction,
                history=history,
            )
            decision.metadata.setdefault("policy", "production-fallback")
            decision.metadata["policy_error"] = str(exc)
            return decision


class CognitiveModule:
    """Lightweight cognitive reasoning delegating intention selection to policies."""

    def __init__(
        self,
        memory_window: int = 8,
        policy: Optional[CognitivePolicy] = None,
    ) -> None:
        self.memory_window = memory_window
        self.episodic_memory: deque[dict[str, Any]] = deque(maxlen=memory_window)
        self.policy: CognitivePolicy = policy or ProductionCognitivePolicy()
        self._confidence_history: deque[float] = deque(maxlen=max(4, memory_window))

    def set_policy(self, policy: CognitivePolicy) -> None:
        """Replace the active policy at runtime."""

        self.policy = policy

    def _summarise_perception(self, perception: PerceptionSnapshot) -> Dict[str, float]:
        summary: Dict[str, float] = {}
        for name, payload in perception.modalities.items():
            spikes = payload.get("spike_counts") or []
            total = float(sum(spikes))
            if total > 0:
                summary[name] = total
        total_sum = sum(summary.values())
        if total_sum <= 0:
            return {name: 0.0 for name in perception.modalities}
        return {name: value / total_sum for name, value in summary.items()}

    def _calibrate_confidence(self, raw_confidence: float) -> float:
        raw = max(0.0, min(1.0, float(raw_confidence)))
        if not self._confidence_history:
            return raw
        mean = sum(self._confidence_history) / len(self._confidence_history)
        if mean <= 0:
            calibrated = raw
        elif raw >= mean:
            calibrated = 0.5 + 0.5 * (raw - mean) / (1 - mean + 1e-6)
        else:
            calibrated = 0.5 * raw / (mean + 1e-6)
        return max(0.0, min(1.0, calibrated))

    def _build_plan(
        self,
        intention: str,
        summary: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        focus = max(summary, key=summary.get) if summary else None
        return default_plan_for_intention(intention, focus, context)

    def _remember(self, summary: Dict[str, float], emotion: EmotionSnapshot, intention: str, confidence: float) -> None:
        self.episodic_memory.append(
            {
                "summary": summary,
                "emotion": emotion.primary.value,
                "intensity": emotion.intensity,
                "intention": intention,
                "confidence": confidence,
            }
        )

    def recall(self, limit: int = 5) -> List[dict[str, Any]]:
        if limit <= 0:
            return list(self.episodic_memory)
        return list(self.episodic_memory)[-limit:]



    def decide(
        self,
        perception: PerceptionSnapshot,
        emotion: EmotionSnapshot,
        personality: PersonalityProfile,
        curiosity: CuriosityState,
        learning_prediction: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = context or {}
        summary = self._summarise_perception(perception)
        try:
            policy_decision = self.policy.select_intention(
                perception,
                summary,
                emotion,
                personality,
                curiosity,
                context,
                learning_prediction,
                history=list(self.episodic_memory),
            )
        except Exception as exc:  # pragma: no cover - defensive policy fallback
            logger.warning("Cognitive policy failed; using fallback plan. Error: %s", exc)
            intention = context.get("fallback_intention", "observe")
            policy_decision = CognitiveDecision(
                intention=intention,
                confidence=0.25,
                plan=self._build_plan(intention, summary, context),
                weights={intention: 1.0},
                tags=[intention, "policy-fallback"],
                focus=max(summary, key=summary.get) if summary else None,
                summary=", ".join(f"{k}:{v:.2f}" for k, v in summary.items())
                or "no-salient-modalities",
                thought_trace=["policy=fallback"],
                perception_summary=dict(summary),
                metadata={"policy": "fallback", "error": str(exc)},
            )

        if not policy_decision.plan:
            policy_decision.plan = self._build_plan(
                policy_decision.intention, summary, context
            )
        if not policy_decision.perception_summary:
            policy_decision.perception_summary = dict(summary)
        if not policy_decision.summary:
            policy_decision.summary = (
                ", ".join(f"{k}:{v:.2f}" for k, v in summary.items())
                or "no-salient-modalities"
            )

        calibrated_confidence = self._calibrate_confidence(policy_decision.confidence)
        policy_decision.confidence = calibrated_confidence
        if policy_decision.focus is None and policy_decision.plan:
            policy_decision.focus = policy_decision.plan[0]

        tags = list(dict.fromkeys(policy_decision.tags)) if policy_decision.tags else []
        if policy_decision.focus and f"focus-{policy_decision.focus}" not in tags:
            tags.append(f"focus-{policy_decision.focus}")
        policy_decision.tags = tags

        self._remember(
            policy_decision.perception_summary,
            emotion,
            policy_decision.intention,
            policy_decision.confidence,
        )
        self._confidence_history.append(policy_decision.confidence)
        policy_decision.metadata.setdefault("confidence_calibrated", True)

        decision = {
            "intention": policy_decision.intention,
            "plan": list(policy_decision.plan),
            "confidence": policy_decision.confidence,
            "weights": dict(policy_decision.weights),
            "tags": list(policy_decision.tags),
            "focus": policy_decision.focus or policy_decision.intention,
            "summary": policy_decision.summary,
            "thought_trace": list(policy_decision.thought_trace),
            "perception_summary": dict(policy_decision.perception_summary),
            "policy_metadata": dict(policy_decision.metadata),
        }
        return decision




__all__ = [
    "default_plan_for_intention",
    "CognitiveDecision",
    "CognitivePolicy",
    "HeuristicCognitivePolicy",
    "StructuredPlanner",
    "ReinforcementCognitivePolicy",
    "BanditCognitivePolicy",
    "ProductionCognitivePolicy",
    "CognitiveModule",
]


