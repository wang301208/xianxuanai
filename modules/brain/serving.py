from __future__ import annotations

"""RPC helpers for delegating brain inference to external serving clusters."""

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from modules.brain.message_bus import publish_neural_event
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
from modules.skills.rpc_client import (
    SkillRPCClient,
    SkillRPCError,
    SkillRPCResponseError,
    SkillRPCTransportError,
)
from schemas.emotion import EmotionType

logger = logging.getLogger(__name__)


class BrainServingError(RuntimeError):
    """Raised when remote brain inference fails."""


@dataclass
class BrainServingResponse:
    status: str
    cycle: BrainCycleResult | None
    metrics: Dict[str, float]
    job_id: str | None = None
    events: Sequence[Dict[str, Any]] | None = None
    raw: Any = None


def _coerce_emotion(payload: Mapping[str, Any]) -> EmotionSnapshot:
    primary = payload.get("primary", EmotionType.NEUTRAL)
    if not isinstance(primary, EmotionType):
        try:
            primary = EmotionType(str(primary).lower())
        except Exception:
            primary = EmotionType.NEUTRAL
    intensity = float(payload.get("intensity", 0.0))
    mood = float(payload.get("mood", 0.0))
    dimensions = {str(k): float(v) for k, v in dict(payload.get("dimensions", {})).items()}
    context = {str(k): float(v) for k, v in dict(payload.get("context", {})).items()}
    decay = float(payload.get("decay", 0.0))
    intent_bias = {str(k): float(v) for k, v in dict(payload.get("intent_bias", {})).items()}
    return EmotionSnapshot(
        primary=primary,
        intensity=intensity,
        mood=mood,
        dimensions=dimensions,
        context=context,
        decay=decay,
        intent_bias=intent_bias,
    )


def _coerce_personality(payload: Mapping[str, Any]) -> PersonalityProfile:
    return PersonalityProfile(
        openness=float(payload.get("openness", 0.5)),
        conscientiousness=float(payload.get("conscientiousness", 0.5)),
        extraversion=float(payload.get("extraversion", 0.5)),
        agreeableness=float(payload.get("agreeableness", 0.5)),
        neuroticism=float(payload.get("neuroticism", 0.5)),
    )


def _coerce_curiosity(payload: Mapping[str, Any]) -> CuriosityState:
    return CuriosityState(
        drive=float(payload.get("drive", 0.4)),
        novelty_preference=float(payload.get("novelty_preference", 0.5)),
        fatigue=float(payload.get("fatigue", 0.1)),
        last_novelty=float(payload.get("last_novelty", 0.0)),
    )


def _coerce_intent(payload: Mapping[str, Any]) -> CognitiveIntent:
    intention = str(payload.get("intention") or "clarify_objective")
    salience = bool(payload.get("salience", False))
    plan = list(payload.get("plan", []) or [])
    confidence = float(payload.get("confidence", 0.5))
    weights = {str(k): float(v) for k, v in dict(payload.get("weights", {})).items()}
    tags = [str(tag) for tag in payload.get("tags", []) or []]
    return CognitiveIntent(
        intention=intention,
        salience=salience,
        plan=list(plan),
        confidence=confidence,
        weights=weights,
        tags=tags,
    )


def _coerce_thought(payload: Mapping[str, Any]) -> ThoughtSnapshot:
    return ThoughtSnapshot(
        focus=str(payload.get("focus") or ""),
        summary=str(payload.get("summary") or ""),
        plan=list(payload.get("plan", []) or []),
        confidence=float(payload.get("confidence", 0.5)),
        memory_refs=list(payload.get("memory_refs", []) or []),
        tags=[str(tag) for tag in payload.get("tags", []) or []],
    )


def _coerce_feeling(payload: Mapping[str, Any]) -> FeelingSnapshot:
    return FeelingSnapshot(
        descriptor=str(payload.get("descriptor") or ""),
        valence=float(payload.get("valence", 0.0)),
        arousal=float(payload.get("arousal", 0.0)),
        mood=float(payload.get("mood", 0.0)),
        confidence=float(payload.get("confidence", 0.0)),
        context_tags=[str(tag) for tag in payload.get("context_tags", []) or []],
    )


def brain_cycle_from_mapping(payload: Mapping[str, Any]) -> BrainCycleResult:
    perception_payload = payload.get("perception", {})
    perception = PerceptionSnapshot(
        modalities=dict(perception_payload.get("modalities", {}) or {}),
        semantic=dict(perception_payload.get("semantic", {}) or {}),
        knowledge_facts=list(perception_payload.get("knowledge_facts", []) or []),
    )

    emotion_payload = payload.get("emotion", {})
    emotion = _coerce_emotion(emotion_payload if isinstance(emotion_payload, Mapping) else {})

    intent_payload = payload.get("intent", {})
    intent = _coerce_intent(intent_payload if isinstance(intent_payload, Mapping) else {})

    personality_payload = payload.get("personality", {})
    personality = _coerce_personality(
        personality_payload if isinstance(personality_payload, Mapping) else {}
    )

    curiosity_payload = payload.get("curiosity", {})
    curiosity = _coerce_curiosity(
        curiosity_payload if isinstance(curiosity_payload, Mapping) else {}
    )

    thoughts_payload = payload.get("thoughts")
    thoughts = None
    if isinstance(thoughts_payload, Mapping):
        thoughts = _coerce_thought(thoughts_payload)

    feeling_payload = payload.get("feeling")
    feeling = None
    if isinstance(feeling_payload, Mapping):
        feeling = _coerce_feeling(feeling_payload)

    metrics = {str(k): float(v) for k, v in dict(payload.get("metrics", {})).items()}
    metadata_payload = payload.get("metadata", {})
    metadata = dict(metadata_payload) if isinstance(metadata_payload, Mapping) else {}

    return BrainCycleResult(
        perception=perception,
        emotion=emotion,
        intent=intent,
        personality=personality,
        curiosity=curiosity,
        energy_used=int(payload.get("energy_used", 0) or 0),
        idle_skipped=int(payload.get("idle_skipped", 0) or 0),
        thoughts=thoughts,
        feeling=feeling,
        metrics=metrics,
        metadata=metadata,
    )


def brain_cycle_to_mapping(cycle: BrainCycleResult) -> Dict[str, Any]:
    payload = asdict(cycle)
    emotion = payload.get("emotion")
    if isinstance(emotion, dict) and isinstance(emotion.get("primary"), EmotionType):
        emotion["primary"] = emotion["primary"].value
    return payload


class BrainServingClient:
    """High-level RPC client coordinating remote brain inference."""

    def __init__(
        self,
        *,
        config,
        rpc_client: SkillRPCClient | None = None,
        logger_obj: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._logger = logger_obj or logger
        defaults = getattr(config, "rpc_defaults", lambda: {})()
        routing_config = getattr(config, "rpc_config", lambda: {})()
        model_name = getattr(config, "model_name", None) or "transformer-brain"
        routing = {model_name: routing_config} if routing_config else {}
        self._client = rpc_client or SkillRPCClient(
            routing=routing,
            defaults=defaults or None,
            default_timeout=float(getattr(config, "timeout", 15.0) or 15.0),
        )
        self._model_name = model_name
        self._metrics_topic = getattr(config, "metrics_topic", None)
        self._result_topic = getattr(config, "result_topic", None)
        self._prefer_batch = bool(getattr(config, "prefer_batch", False))

    @classmethod
    def from_config(cls, config, **kwargs) -> "BrainServingClient":
        return cls(config=config, **kwargs)

    def infer(
        self,
        inputs: Mapping[str, Any],
        *,
        model_name: Optional[str] = None,
        batch: Optional[Sequence[Mapping[str, Any]]] = None,
        context: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> BrainServingResponse:
        target_model = model_name or self._model_name
        payload: Dict[str, Any] = {"inputs": dict(inputs)}
        batch_size = None
        if batch:
            payload["batch"] = [dict(item) for item in batch]
            batch_size = len(batch)
        elif self._prefer_batch:
            payload["batch"] = [dict(inputs)]
            batch_size = 1

        metadata = {"execution_mode": "rpc", "model": target_model}
        request_meta = getattr(self._config, "to_request_metadata", None)
        if callable(request_meta):
            metadata.update(request_meta(batch_size=batch_size))
        rpc_config = getattr(self._config, "rpc_config", None)
        if callable(rpc_config):
            metadata["rpc_config"] = rpc_config()

        request_timeout = timeout or float(getattr(self._config, "timeout", 15.0) or 15.0)
        try:
            raw = self._client.invoke(
                target_model,
                payload,
                context=context or {},
                metadata=metadata,
                timeout=request_timeout,
            )
        except (SkillRPCResponseError, SkillRPCTransportError) as exc:
            self._logger.warning("Brain serving transport failure: %s", exc)
            raise BrainServingError(str(exc)) from exc
        except SkillRPCError as exc:
            self._logger.error("Brain serving invocation failed: %s", exc, exc_info=True)
            raise BrainServingError(str(exc)) from exc

        response = self._parse_response(raw)
        self._dispatch_side_effects(response, model_name=target_model)
        return response

    def _parse_response(self, raw: Any) -> BrainServingResponse:
        if not isinstance(raw, Mapping):
            return BrainServingResponse(status="ok", cycle=None, metrics={}, raw=raw)

        status = str(raw.get("status") or "ok")
        metrics = raw.get("metrics") or raw.get("telemetry") or {}
        metrics_payload = {
            str(k): float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        job_id = raw.get("job_id") or raw.get("task_id")
        events_payload = raw.get("events")
        if isinstance(events_payload, list):
            events = events_payload
        elif isinstance(events_payload, tuple):
            events = list(events_payload)
        else:
            events = None
        cycle_payload: Mapping[str, Any] | None = None
        for key in ("result", "cycle", "output", "outputs"):
            candidate = raw.get(key)
            if isinstance(candidate, Mapping) and {"intent", "emotion"}.issubset(candidate.keys()):
                cycle_payload = candidate
                break
        if cycle_payload is None and {"intent", "emotion"}.issubset(raw.keys()):
            cycle_payload = raw

        cycle = None
        if cycle_payload is not None:
            try:
                cycle = brain_cycle_from_mapping(cycle_payload)
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.warning("Failed to decode brain cycle payload: %s", exc, exc_info=True)

        return BrainServingResponse(
            status=status,
            cycle=cycle,
            metrics=metrics_payload,
            job_id=str(job_id) if job_id else None,
            events=list(events) if isinstance(events, Sequence) else None,
            raw=raw,
        )

    def _dispatch_side_effects(self, response: BrainServingResponse, *, model_name: str) -> None:
        if response.metrics and self._metrics_topic:
            event = {
                "target": self._metrics_topic,
                "source": "brain-serving",
                "model": model_name,
                "metrics": response.metrics,
                "status": response.status,
            }
            try:
                publish_neural_event(event)
            except Exception:  # pragma: no cover - best effort
                self._logger.debug("Failed to publish brain metrics event.", exc_info=True)

        if response.cycle and self._result_topic:
            event = {
                "target": self._result_topic,
                "model": model_name,
                "cycle": brain_cycle_to_mapping(response.cycle),
                "status": response.status,
            }
            try:
                publish_neural_event(event)
            except Exception:  # pragma: no cover - best effort
                self._logger.debug("Failed to publish brain result event.", exc_info=True)

        if response.events:
            for raw_event in response.events:
                if not isinstance(raw_event, Mapping):
                    continue
                event = dict(raw_event)
                if "target" not in event:
                    continue
                try:
                    publish_neural_event(event)  # type: ignore[arg-type]
                except Exception:  # pragma: no cover - best effort
                    self._logger.debug("Failed to publish auxiliary brain event.", exc_info=True)


__all__ = [
    "BrainServingClient",
    "BrainServingError",
    "BrainServingResponse",
    "brain_cycle_from_mapping",
    "brain_cycle_to_mapping",
]
