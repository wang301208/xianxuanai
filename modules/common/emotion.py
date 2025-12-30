"""Utilities for analyzing and tracking emotional state in dialogues.

This module provides a pluggable architecture for emotion analysis through
the :class:`BaseEmotionModel` interface.  A lightweight
 :class:`KeywordModel` implements a simple keyword heuristic while
 :class:`MLModel` acts as a placeholder for more advanced machine‑learning
 approaches.  An :class:`EmotionAnalyzer` facade exposes a uniform API for
 these models.

An :class:`EmotionState` container tracks conversational mood and the helper
:func:`adjust_response_style` can adapt responses to detected emotions.  The
optional :class:`EmotionProfile` allows culture or personality specific
thresholds and response styles.  Voice analysis hooks are present so a
speech‑to‑text pipeline can be integrated in the future.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Protocol

try:  # pragma: no cover - import guard for optional dependency
    from transformers import pipeline
except ModuleNotFoundError:  # pragma: no cover
    pipeline = None  # type: ignore


# ---------------------------------------------------------------------------
# Profiles


@dataclass
class EmotionProfile:
    """Customize emotion classification thresholds and response style.

    Parameters
    ----------
    positive_threshold:
        Difference in positive vs. negative keyword counts required to label a
        message as positive.
    negative_threshold:
        Difference in negative vs. positive keyword counts required to label a
        message as negative.
    positive_suffix:
        Suffix appended to positive responses.
    negative_prefix:
        Prefix prepended to negative responses.
    """

    positive_threshold: int = 1
    negative_threshold: int = 1
    positive_suffix: str = " \U0001F60A"  # smiling face emoji
    negative_prefix: str = "I'm sorry to hear that."


# ---------------------------------------------------------------------------
# Model interface and implementations


class BaseEmotionModel(Protocol):
    """Protocol for pluggable emotion classification models."""

    def analyze_text(self, text: str, profile: EmotionProfile | None = None) -> str:
        """Return an emotion label for *text*."""

    def analyze_voice(self, audio_bytes: bytes, profile: EmotionProfile | None = None) -> str:
        """Return an emotion label for *audio_bytes*."""


def _speech_to_text(audio_bytes: bytes) -> str:
    """Placeholder speech‑to‑text pipeline.

    The current implementation simply decodes the bytes as UTF‑8, ignoring
    errors.  Replace this function with a call to a real speech‑to‑text model
    when integrating with an audio processing library.
    """

    return audio_bytes.decode("utf-8", errors="ignore")


class KeywordModel(BaseEmotionModel):
    """Classify emotions using simple keyword heuristics."""

    POSITIVE_KEYWORDS = {
        "good",
        "great",
        "happy",
        "love",
        "excellent",
        "awesome",
        "fantastic",
        "amazing",
    }

    NEGATIVE_KEYWORDS = {
        "bad",
        "sad",
        "angry",
        "hate",
        "terrible",
        "awful",
        "horrible",
        "worse",
    }

    def analyze_text(self, text: str, profile: EmotionProfile | None = None) -> str:
        profile = profile or EmotionProfile()
        lowered = text.lower()
        pos = sum(word in lowered for word in self.POSITIVE_KEYWORDS)
        neg = sum(word in lowered for word in self.NEGATIVE_KEYWORDS)
        if pos - neg >= profile.positive_threshold:
            return "positive"
        if neg - pos >= profile.negative_threshold:
            return "negative"
        return "neutral"

    def analyze_voice(self, audio_bytes: bytes, profile: EmotionProfile | None = None) -> str:
        text = _speech_to_text(audio_bytes)
        return self.analyze_text(text, profile)


class MLModel(BaseEmotionModel):  # pragma: no cover - placeholder
    """ML based emotion classification using pretrained HuggingFace pipelines.

    Parameters
    ----------
    text_model:
        HuggingFace model identifier for text classification.  Defaults to a
        sentiment model if not provided.
    audio_model:
        Optional audio classification model identifier.  When provided, the
        model is used directly on audio bytes without transcription.
    asr_model:
        Optional speech‑to‑text model identifier.  When supplied, audio is
        transcribed before text classification; otherwise a lightweight UTF‑8
        decode fallback is used.
    device:
        Device string or integer understood by :func:`transformers.pipeline`
        (e.g., ``"cpu"`` or CUDA device index).
    label_mapping:
        Mapping from model output labels to the standard ``positive``,
        ``negative``, or ``neutral`` categories expected by the rest of the
        system.  Unmapped labels are lower‑cased.
    """

    def __init__(
        self,
        text_model: str | None = None,
        audio_model: str | None = None,
        asr_model: str | None = None,
        device: int | str | None = None,
        label_mapping: dict[str, str] | None = None,
    ) -> None:
        self.text_model_name = text_model or "distilbert-base-uncased-finetuned-sst-2-english"
        self.audio_model_name = audio_model
        self.asr_model_name = asr_model
        self.device = device
        self.label_mapping = label_mapping or {
            "POSITIVE": "positive",
            "NEGATIVE": "negative",
            "NEUTRAL": "neutral",
            "positive": "positive",
            "negative": "negative",
            "neutral": "neutral",
        }
        self._text_classifier = None
        self._audio_classifier = None
        self._asr = None

    def _require_pipeline(self) -> None:
        if pipeline is None:
            raise ImportError(
                "transformers is required for MLModel; please install it or provide a mocked pipeline"
            )

    def _get_text_classifier(self):
        self._require_pipeline()
        if self._text_classifier is None:
            self._text_classifier = pipeline(
                "text-classification", model=self.text_model_name, device=self.device
            )
        return self._text_classifier

    def _get_audio_classifier(self):
        self._require_pipeline()
        if self.audio_model_name and self._audio_classifier is None:
            self._audio_classifier = pipeline(
                "audio-classification", model=self.audio_model_name, device=self.device
            )
        return self._audio_classifier

    def _get_asr(self):
        self._require_pipeline()
        if self.asr_model_name and self._asr is None:
            self._asr = pipeline(
                "automatic-speech-recognition", model=self.asr_model_name, device=self.device
            )
        return self._asr

    def _map_label(self, label: str) -> str:
        return self.label_mapping.get(label, self.label_mapping.get(label.lower(), label.lower()))

    def analyze_text(self, text: str, _profile: EmotionProfile | None = None) -> str:
        classifier = self._get_text_classifier()
        result = classifier(text)[0]
        return self._map_label(result.get("label", "neutral"))

    def _speech_to_text(self, audio_bytes: bytes) -> str:
        asr = self._get_asr()
        if asr:
            result = asr(audio_bytes)
            if isinstance(result, dict):
                return str(result.get("text", ""))
            return str(result)
        return _speech_to_text(audio_bytes)

    def analyze_voice(self, audio_bytes: bytes, _profile: EmotionProfile | None = None) -> str:
        audio_classifier = self._get_audio_classifier()
        if audio_classifier:
            result = audio_classifier(audio_bytes)[0]
            return self._map_label(result.get("label", "neutral"))

        transcript = self._speech_to_text(audio_bytes)
        return self.analyze_text(transcript, _profile)


# ---------------------------------------------------------------------------
# Analyzer facade and state


class EmotionAnalyzer:
    """Facade providing a uniform interface over concrete emotion models."""

    def __init__(self, model: BaseEmotionModel | None = None, profile: EmotionProfile | None = None) -> None:
        self.model = model or KeywordModel()
        self.profile = profile

    def analyze_text(self, text: str) -> str:
        return self.model.analyze_text(text, self.profile)

    def analyze_voice(self, audio_bytes: bytes) -> str:
        return self.model.analyze_voice(audio_bytes, self.profile)


@dataclass
class EmotionState:
    """Represents the tracked emotional state of the conversation."""

    label: str = "neutral"
    excitement: float = 0.0
    satisfaction: float = 0.0

    def update(self, text: str, analyzer: EmotionAnalyzer) -> None:
        """Update state based on *text* classified by *analyzer*."""

        self.label = analyzer.analyze_text(text)
        if self.label == "positive":
            self.excitement = min(1.0, self.excitement + 0.1)
            self.satisfaction = min(1.0, self.satisfaction + 0.1)
        elif self.label == "negative":
            self.excitement = max(0.0, self.excitement - 0.1)
            self.satisfaction = max(0.0, self.satisfaction - 0.1)


# ---------------------------------------------------------------------------
# Response adjustment


def adjust_response_style(
    response: str,
    state: EmotionState,
    profile: EmotionProfile | None = None,
    signals: Iterable[str] | None = None,
) -> str:
    """Return *response* adapted to the current emotion and extra signals.

    Parameters
    ----------
    response:
        The base response text to adjust.
    state:
        The :class:`EmotionState` used as the primary signal.
    profile:
        Optional :class:`EmotionProfile` controlling thresholds and style.
    signals:
        Additional emotion labels from other modalities (e.g., voice).  The
        final label is the majority among ``state.label`` and these signals.
    """

    all_labels = [state.label]
    if signals:
        all_labels.extend(signals)
    label = Counter(all_labels).most_common(1)[0][0]

    profile = profile or EmotionProfile()
    if label == "positive":
        return f"{response}{profile.positive_suffix}"
    if label == "negative":
        return f"{profile.negative_prefix} {response}".strip()
    return response


__all__ = [
    "BaseEmotionModel",
    "KeywordModel",
    "MLModel",
    "EmotionAnalyzer",
    "EmotionProfile",
    "EmotionState",
    "adjust_response_style",
]

