"""
Factories and adapters that expose different brain backends through a common API.

The AutoGPT agent stack already knows how to talk to
``WholeBrainSimulation`` via :class:`modules.brain.adapter.WholeBrainAgentAdapter`.
This module extends that capability so that other cognitive backends—
most notably :mod:`BrainSimulationSystem`—can be slotted in without
rewriting the agent loop.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, MutableMapping, Sequence

from third_party.autogpt.autogpt.core.brain.config import (
    BrainBackend,
    BrainSimulationConfig,
    WholeBrainConfig,
)
from modules.brain.backends.base import BrainBackendProtocol
from modules.brain.state import (
    BrainCycleResult,
    CognitiveIntent,
    CuriosityState,
    EmotionSnapshot,
    FeelingSnapshot,
    PerceptionSnapshot,
    PersonalityProfile,
    ThoughtSnapshot,
)
from schemas.emotion import EmotionType

logger = logging.getLogger(__name__)


class BrainBackendInitError(RuntimeError):
    """Raised when a brain backend cannot be initialised."""


def create_brain_backend(
    backend: BrainBackend,
    *,
    whole_brain_config: WholeBrainConfig,
    brain_simulation_config: BrainSimulationConfig,
) -> BrainBackendProtocol:
    """
    Instantiate the requested backend.

    Args:
        backend: Enum describing which implementation should be used.
        whole_brain_config: Configuration passed to ``WholeBrainSimulation``.
        brain_simulation_config: Configuration passed to ``BrainSimulationSystem``.

    Returns:
        Object exposing ``process_cycle`` compatible with ``WholeBrainAgentAdapter``.
    """

    if backend == BrainBackend.WHOLE_BRAIN:
        return _create_whole_brain_backend(whole_brain_config)
    if backend == BrainBackend.BRAIN_SIMULATION:
        return BrainSimulationSystemAdapter(brain_simulation_config)
    raise BrainBackendInitError(f"Unsupported brain backend '{backend}'.")


def _create_whole_brain_backend(
    whole_brain_config: WholeBrainConfig,
) -> BrainBackendProtocol:
    brain_kwargs = whole_brain_config.to_simulation_kwargs()
    from modules.brain.whole_brain import WholeBrainSimulation

    return WholeBrainSimulation(**brain_kwargs)


class BrainSimulationSystemAdapter:
    """Adapter that embeds ``BrainSimulationSystem`` behind the WholeBrain API."""

    def __init__(self, config: BrainSimulationConfig) -> None:
        payload = config.to_backend_payload()
        try:
            from BrainSimulationSystem.brain_simulation import BrainSimulation
        except Exception as exc:  # pragma: no cover - optional dependency
            raise BrainBackendInitError(
                "BrainSimulationSystem is unavailable; ensure dependencies are installed."
            ) from exc

        overrides = payload.get("overrides") or {}
        try:
            try:
                self._brain = BrainSimulation(
                    overrides or None,
                    profile=payload.get("profile"),
                    stage=payload.get("stage"),
                )
            except TypeError:
                self._brain = BrainSimulation(overrides or None)
        except Exception as exc:  # pragma: no cover - adaptor start-up
            raise BrainBackendInitError(
                "Failed to initialise BrainSimulationSystem"
            ) from exc

        self._dt = float(payload.get("dt", 100.0))
        self._metadata = {
            "profile": payload.get("profile"),
            "stage": payload.get("stage"),
        }
        if payload.get("auto_background"):
            try:
                self._brain.start_continuous_simulation(self._dt)
            except Exception:  # pragma: no cover - non-critical best effort
                logger.debug("Background brain loop failed to start.", exc_info=True)

    def process_cycle(self, input_payload: Mapping[str, Any]) -> BrainCycleResult:
        BrainSimulationSystemAdapter._validate_brain()
        sim_inputs = translate_agent_payload_for_brain_simulation(input_payload)
        sim_inputs.setdefault("metadata", {}).update(self._metadata)
        raw_result = self._brain.step(sim_inputs, self._dt)
        return brain_simulation_result_to_cycle(raw_result)

    def shutdown(self) -> None:
        if hasattr(self._brain, "stop_continuous_simulation"):
            try:
                self._brain.stop_continuous_simulation()
            except Exception:  # pragma: no cover - best effort
                logger.debug("Failed to stop BrainSimulationSystem background loop.", exc_info=True)

    def attach_knowledge_base(self, knowledge_base: Any) -> None:  # pragma: no cover - optional hook
        attach = getattr(self._brain, "attach_knowledge_base", None)
        if callable(attach):
            try:
                attach(knowledge_base)
            except TypeError:
                attach(knowledge_base=knowledge_base)
            return

        try:
            setattr(self._brain, "knowledge_base", knowledge_base)
        except Exception:
            logger.debug("Unable to attach knowledge base to BrainSimulationSystem.", exc_info=True)

    def update_config(
        self,
        runtime_config: Any | None = None,
        *,
        overrides: Mapping[str, Any] | None = None,
    ) -> None:  # pragma: no cover - optional hook
        if overrides:
            if hasattr(self._brain, "update_parameters"):
                try:
                    self._brain.update_parameters(dict(overrides))
                except Exception:
                    logger.debug(
                        "BrainSimulationSystem.update_parameters failed.",
                        exc_info=True,
                    )
            return
        if runtime_config is not None and hasattr(self._brain, "update_parameters"):
            try:
                self._brain.update_parameters({"runtime": runtime_config})
            except Exception:
                logger.debug(
                    "BrainSimulationSystem.update_parameters failed.", exc_info=True
                )

    def __getattr__(self, item: str) -> Any:
        """Expose selected attributes of the wrapped simulation (best effort)."""

        return getattr(self._brain, item)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"BrainSimulationSystemAdapter(profile={self._metadata.get('profile')!r})"

    @staticmethod
    def _validate_brain() -> None:
        """Placeholder for future health checks."""
        return


def translate_agent_payload_for_brain_simulation(
    agent_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Project WholeBrain inputs onto the richer BrainSimulationSystem schema."""

    context = _ensure_mapping(agent_payload.get("context"))
    sensory = {
        "vision": _ensure_numeric_vector(agent_payload.get("vision")),
        "auditory": _ensure_numeric_vector(agent_payload.get("auditory")),
        "somatosensory": _ensure_numeric_vector(agent_payload.get("somatosensory")),
        "text": agent_payload.get("text"),
    }
    sensory_data: list[float] = []
    for channel in ("vision", "auditory", "somatosensory"):
        values = sensory.get(channel) or []
        if isinstance(values, list):
            sensory_data.extend([float(v) for v in values])
    goals: list[str] = []
    if "goal_focus" in context and context["goal_focus"]:
        goals.append(str(context["goal_focus"]))
    task = agent_payload.get("task")
    if isinstance(task, str) and task:
        goals.append(task)

    emotion_stimuli = _build_emotion_stimuli(context)
    curiosity_payload = {
        "novelty": _safe_float(context.get("novelty"), 0.5),
        "progress": _safe_float(context.get("progress"), 0.0),
        "threat": _safe_float(context.get("threat"), 0.0),
        "safety": _safe_float(context.get("safety"), 0.5),
    }

    payload: Dict[str, Any] = {
        "perception": sensory,
        "sensory_data": sensory_data,
        "vision": sensory["vision"],
        "auditory": sensory["auditory"],
        "somatosensory": sensory["somatosensory"],
        "language_input": sensory.get("text"),
        "goals": goals[:5],
        "task": task,
        "attention_bias": {"salient": bool(agent_payload.get("is_salient"))},
        "curiosity": curiosity_payload,
        "emotion": {"stimuli": emotion_stimuli},
        "memory_context": {
            "context": context,
            "history": agent_payload.get("text"),
        },
        "reward": _safe_float(context.get("progress"), 0.0),
        "metadata": {"agent_id": agent_payload.get("agent_id")},
    }
    return payload


def brain_simulation_result_to_cycle(payload: Mapping[str, Any]) -> BrainCycleResult:
    """Translate BrainSimulationSystem output into ``BrainCycleResult``."""

    cognitive = _ensure_mapping(payload.get("cognitive_state"))
    network_state = _ensure_mapping(payload.get("network_state"))

    perception_snapshot = _build_perception_snapshot(_ensure_mapping(cognitive.get("perception")))
    emotion_snapshot = _build_emotion_snapshot(_ensure_mapping(cognitive.get("emotion")))
    curiosity = _build_curiosity_snapshot(_ensure_mapping(cognitive.get("curiosity")))
    personality = _build_personality_snapshot(_ensure_mapping(cognitive.get("personality")))
    intent = _build_cognitive_intent(
        _ensure_mapping(cognitive.get("decision")),
        _ensure_mapping(cognitive.get("planner")),
    )
    thoughts = _build_thought_snapshot(cognitive, intent)
    feeling = _build_feeling_snapshot(emotion_snapshot)

    energy_used = _estimate_energy(network_state)
    metrics = _collect_metrics(cognitive, network_state)
    metadata = {
        "backend": "brain_simulation",
        "time": payload.get("time"),
    }

    return BrainCycleResult(
        perception=perception_snapshot,
        emotion=emotion_snapshot,
        intent=intent,
        personality=personality,
        curiosity=curiosity,
        energy_used=energy_used,
        idle_skipped=0,
        thoughts=thoughts,
        feeling=feeling,
        metrics=metrics,
        metadata=metadata,
    )


def _build_emotion_stimuli(context: Mapping[str, Any]) -> list[Dict[str, Any]]:
    payload: list[Dict[str, Any]] = []
    for key in ("success_rate", "failure_rate", "progress", "novelty", "threat"):
        value = context.get(key)
        if isinstance(value, (int, float)):
            payload.append({"type": key, "intensity": float(value)})
    return payload or [{"type": "baseline", "intensity": 0.5}]


def _build_perception_snapshot(perception_state: Mapping[str, Any]) -> PerceptionSnapshot:
    semantic = _ensure_mapping(
        perception_state.get("semantic")
        or perception_state.get("semantic_map")
        or {}
    )
    modalities = _ensure_mapping(perception_state.get("modalities") or {})
    fused_embedding = _ensure_numeric_vector(perception_state.get("fused_embedding"))
    modality_embeddings: Dict[str, list[float]] = {}
    embeddings_state = _ensure_mapping(perception_state.get("modality_embeddings") or {})
    for channel, values in embeddings_state.items():
        modality_embeddings[str(channel)] = _ensure_numeric_vector(values)

    knowledge_facts: list[Dict[str, Any]] = []
    knowledge_updates = perception_state.get("knowledge_updates")
    if isinstance(knowledge_updates, Mapping):
        added = knowledge_updates.get("added")
        if isinstance(added, Sequence):
            knowledge_facts = [dict(fact) for fact in added if isinstance(fact, Mapping)]

    return PerceptionSnapshot(
        modalities=modalities,
        semantic=semantic,
        knowledge_facts=knowledge_facts,
        fused_embedding=fused_embedding,
        modality_embeddings=modality_embeddings,
    )


def _build_emotion_snapshot(emotion_state: Mapping[str, Any]) -> EmotionSnapshot:
    primary_raw = (
        emotion_state.get("primary")
        or emotion_state.get("dominant")
        or emotion_state.get("label")
        or EmotionType.NEUTRAL.value
    )
    primary = _coerce_emotion_type(primary_raw)
    dimensions = _float_map(emotion_state.get("dimensions"))
    context = _float_map(emotion_state.get("context"))
    intent_bias = _float_map(emotion_state.get("intent_bias"))

    return EmotionSnapshot(
        primary=primary,
        intensity=_safe_float(emotion_state.get("intensity"), 0.5),
        mood=_safe_float(emotion_state.get("mood"), 0.5),
        dimensions=dimensions,
        context=context,
        decay=_safe_float(emotion_state.get("decay"), 0.0),
        intent_bias=intent_bias,
    )


def _build_curiosity_snapshot(curiosity_state: Mapping[str, Any]) -> CuriosityState:
    stimulus = _ensure_mapping(curiosity_state.get("stimulus"))
    return CuriosityState(
        drive=_safe_float(curiosity_state.get("drive"), 0.4),
        novelty_preference=_safe_float(stimulus.get("novelty"), 0.5),
        fatigue=_safe_float(curiosity_state.get("fatigue"), 0.1),
        last_novelty=_safe_float(stimulus.get("novelty"), 0.0),
    )


def _build_personality_snapshot(personality_state: Mapping[str, Any]) -> PersonalityProfile:
    profile = PersonalityProfile()
    dynamic = _ensure_mapping(personality_state.get("dynamic") or personality_state.get("traits") or {})
    for trait in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
        if trait in dynamic:
            setattr(profile, trait, _clamp01(_safe_float(dynamic.get(trait), 0.5)))
    return profile


def _build_cognitive_intent(
    decision_state: Mapping[str, Any],
    planner_state: Mapping[str, Any],
) -> CognitiveIntent:
    plan: list[str] = []
    planner_plan = planner_state.get("plan")
    if isinstance(planner_plan, Sequence):
        plan = [str(step) for step in planner_plan if step]
    elif isinstance(planner_state.get("candidates"), Sequence):
        plan = [str(item.get("action")) for item in planner_state["candidates"] if isinstance(item, Mapping) and item.get("action")]

    weights = _float_map(decision_state.get("scores") or decision_state.get("weights"))

    tags: list[str] = []
    raw_tags = decision_state.get("tags") or planner_state.get("tags")
    if isinstance(raw_tags, Sequence):
        tags = [str(tag) for tag in raw_tags if tag]

    return CognitiveIntent(
        intention=str(decision_state.get("decision") or decision_state.get("intent") or "reflect"),
        salience=bool(decision_state.get("salience") or decision_state.get("is_salient")),
        plan=plan,
        confidence=_safe_float(decision_state.get("confidence"), 0.5),
        weights=weights,
        tags=tags,
    )


def _build_thought_snapshot(
    cognitive_state: Mapping[str, Any],
    intent: CognitiveIntent,
) -> ThoughtSnapshot:
    attention = _ensure_mapping(cognitive_state.get("attention_manager"))
    focus = attention.get("focus") or intent.intention
    summary = ""
    notes = _ensure_sequence(_ensure_mapping(cognitive_state.get("decision")).get("notes"))
    if notes:
        summary = str(notes[0])
    elif attention.get("summary"):
        summary = str(attention.get("summary"))

    plan_from_attention = attention.get("plan") if isinstance(attention.get("plan"), Sequence) else None
    plan = intent.plan or ([str(step) for step in plan_from_attention] if plan_from_attention else [])

    return ThoughtSnapshot(
        focus=str(focus or intent.intention),
        summary=summary or intent.intention,
        plan=plan,
        confidence=intent.confidence,
        memory_refs=[],
        tags=intent.tags,
    )


def _build_feeling_snapshot(emotion_snapshot: EmotionSnapshot) -> FeelingSnapshot:
    return FeelingSnapshot(
        descriptor=emotion_snapshot.primary.value,
        valence=emotion_snapshot.context.get("valence", emotion_snapshot.intensity),
        arousal=emotion_snapshot.context.get("arousal", emotion_snapshot.intensity),
        mood=emotion_snapshot.mood,
        confidence=emotion_snapshot.intent_bias.get("confidence", emotion_snapshot.intensity),
        context_tags=list(emotion_snapshot.context.keys()),
    )


def _estimate_energy(network_state: Mapping[str, Any]) -> int:
    spikes = network_state.get("spikes")
    if isinstance(spikes, Sequence):
        return max(len(spikes), 1)
    return 1


def _collect_metrics(
    cognitive_state: Mapping[str, Any],
    network_state: Mapping[str, Any],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    self_supervised = _ensure_mapping(cognitive_state.get("self_supervised"))
    metrics.update(_float_map(self_supervised))
    metrics.update(_float_map(network_state))
    return metrics


def _coerce_emotion_type(value: Any) -> EmotionType:
    if isinstance(value, EmotionType):
        return value
    try:
        return EmotionType(str(value).lower())
    except Exception:
        return EmotionType.NEUTRAL


def _ensure_mapping(value: Any) -> MutableMapping[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _ensure_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return []


def _ensure_numeric_vector(value: Any) -> list[float]:
    if not isinstance(value, Sequence):
        return []
    result: list[float] = []
    for item in value[:32]:
        if isinstance(item, (int, float)):
            result.append(float(item))
    return result


def _float_map(value: Any) -> Dict[str, float]:
    mapping = _ensure_mapping(value)
    floatified: Dict[str, float] = {}
    for key, raw in mapping.items():
        if isinstance(raw, (int, float)):
            floatified[str(key)] = float(raw)
    return floatified


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


__all__ = [
    "BrainBackendProtocol",
    "BrainBackendInitError",
    "BrainSimulationSystemAdapter",
    "create_brain_backend",
    "translate_agent_payload_for_brain_simulation",
    "brain_simulation_result_to_cycle",
]
