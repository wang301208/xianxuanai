"""High level wrapper around the :mod:`modules.brain.limbic` module."""

from __future__ import annotations

import logging
from typing import Dict, Optional

from schemas.emotion import EmotionalState, EmotionType
from modules.brain.emotion_service import EmotionServiceClient, EmotionServiceConfig
from modules.brain.limbic import LimbicSystem
from modules.brain.state import BrainRuntimeConfig

logger = logging.getLogger(__name__)


class EmotionEngine:
    """Facade exposing a simple ``process_emotion`` API."""

    def __init__(
        self,
        limbic_system: LimbicSystem | None = None,
        config: BrainRuntimeConfig | None = None,
        service_client: EmotionServiceClient | None = None,
    ) -> None:
        self.config = config or BrainRuntimeConfig()
        self.limbic_system = limbic_system or LimbicSystem()
        self.service_client = self._init_service_client(service_client)

    def process_emotion(self, stimulus: str, context: Optional[Dict[str, float]] = None) -> EmotionalState:
        """Return an :class:`EmotionalState` derived from ``stimulus``."""

        context = context or {}
        state = self._process_with_service(stimulus, context)
        if state:
            return state
        return self.limbic_system.react(stimulus, context=context, config=self.config)

    def update_config(self, config: BrainRuntimeConfig) -> None:
        """Update runtime configuration for subsequent emotion evaluations."""

        self.config = config

    def _init_service_client(
        self, service_client: EmotionServiceClient | None
    ) -> EmotionServiceClient | None:
        if service_client:
            return service_client
        service_config = getattr(self.config, "emotion_service", None)
        if isinstance(service_config, EmotionServiceConfig) and service_config.enabled:
            try:
                return EmotionServiceClient(service_config)
            except Exception:
                logger.exception("Failed to initialize emotion service client")
                return None
        return None

    def _process_with_service(
        self, stimulus: str, context: Dict[str, float]
    ) -> EmotionalState | None:
        if not self.service_client:
            return None
        try:
            result = self.service_client.analyze(stimulus, context)
        except Exception:
            logger.exception("Emotion service call failed; falling back to local model")
            return None
        return self._map_service_result(result, stimulus, context)

    def _map_service_result(
        self,
        result: Dict[str, object],
        stimulus: str,
        context: Dict[str, float],
    ) -> EmotionalState:
        scores = self._normalize_scores(result)
        emotion = self._resolve_emotion_label(result, scores)
        full_dimensions = self._resolve_dimensions(result, emotion, scores)
        context_weights = dict(context)
        confidence = max(scores.values()) if scores else None
        if confidence is not None:
            context_weights.setdefault("model_confidence", confidence)
        service_context = result.get("context") if isinstance(result, dict) else None
        if isinstance(service_context, dict):
            for key, value in service_context.items():
                try:
                    context_weights.setdefault(key, float(value))
                except (TypeError, ValueError):
                    continue

        return self.limbic_system.build_state(
            emotion,
            full_dimensions,
            context_weights,
            config=self.config,
            stimulus=stimulus,
        )

    def _normalize_scores(self, result: Dict[str, object]) -> Dict[str, float]:
        scores_raw = {}
        if isinstance(result, dict):
            for key in ("scores", "probabilities", "emotions"):
                scores_candidate = result.get(key)
                if isinstance(scores_candidate, dict):
                    scores_raw = scores_candidate
                    break
        normalized: Dict[str, float] = {}
        total = 0.0
        for label, score in scores_raw.items():
            try:
                value = max(0.0, float(score))
            except (TypeError, ValueError):
                continue
            total += value
            normalized[str(label).lower()] = value
        if total > 0:
            return {label: value / total for label, value in normalized.items()}
        return normalized

    def _resolve_emotion_label(
        self, result: Dict[str, object], scores: Dict[str, float]
    ) -> EmotionType:
        label = None
        if isinstance(result, dict):
            for key in ("label", "emotion", "primary"):
                value = result.get(key)
                if isinstance(value, str):
                    label = value
                    break
        if not label and scores:
            label = max(scores.items(), key=lambda item: item[1])[0]
        return self._map_label_to_emotion(label)

    def _map_label_to_emotion(self, label: str | None) -> EmotionType:
        if not label:
            return EmotionType.NEUTRAL
        normalized = label.lower()
        mapping = {
            "joy": EmotionType.HAPPY,
            "positive": EmotionType.HAPPY,
            "happiness": EmotionType.HAPPY,
            "happy": EmotionType.HAPPY,
            "anger": EmotionType.ANGRY,
            "angry": EmotionType.ANGRY,
            "mad": EmotionType.ANGRY,
            "annoyed": EmotionType.ANGRY,
            "sad": EmotionType.SAD,
            "sadness": EmotionType.SAD,
            "negative": EmotionType.SAD,
            "depressed": EmotionType.SAD,
            "neutral": EmotionType.NEUTRAL,
        }
        return mapping.get(normalized, EmotionType.NEUTRAL)

    def _resolve_dimensions(
        self,
        result: Dict[str, object],
        emotion: EmotionType,
        scores: Dict[str, float],
    ) -> Dict[str, float]:
        dimensions: Dict[str, float] = {}
        if isinstance(result, dict):
            dims_candidate = result.get("dimensions")
            if isinstance(dims_candidate, dict):
                for key in ("valence", "arousal", "dominance"):
                    try:
                        dimensions[key] = float(dims_candidate.get(key, dimensions.get(key, 0.0)))
                    except (TypeError, ValueError):
                        continue
        if dimensions:
            return dimensions

        defaults = {
            EmotionType.HAPPY: {"valence": 0.65, "arousal": 0.55, "dominance": 0.35},
            EmotionType.SAD: {"valence": -0.55, "arousal": 0.35, "dominance": -0.25},
            EmotionType.ANGRY: {"valence": -0.65, "arousal": 0.70, "dominance": 0.15},
            EmotionType.NEUTRAL: {"valence": 0.0, "arousal": 0.35, "dominance": 0.05},
        }
        base_dimensions = defaults.get(emotion, defaults[EmotionType.NEUTRAL])
        confidence = max(scores.values()) if scores else 0.5
        scaled = dict(base_dimensions)
        scaled["valence"] = max(-1.0, min(1.0, scaled["valence"] * (0.5 + confidence)))
        scaled["arousal"] = max(0.0, min(1.0, scaled["arousal"] * (0.5 + confidence)))
        scaled["dominance"] = max(-1.0, min(1.0, scaled["dominance"] * (0.5 + confidence)))
        return scaled


__all__ = ["EmotionEngine"]

