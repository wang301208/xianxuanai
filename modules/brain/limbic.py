"""Data-driven limbic system components for emotion processing.

This module provides deployable substitutes for brain regions typically
associated with emotional processing.  The goal is to keep the API
practical for large-scale agents while implementing multi-dimensional
valence–arousal–dominance modelling and homeostatic regulation hooks that
are exercised throughout the cognitive stack.

The :class:`LimbicSystem` orchestrates three sub-modules:

``EmotionProcessor``
    Derives a primary emotion from a textual stimulus.
``MemoryConsolidator``
    Stores past stimulus/response pairs for later inspection.
``HomeostasisController``
    Keeps the emotional intensity within a bounded range while adjusting mood.
"""

from __future__ import annotations

import hashlib
import math
import re
from statistics import StatisticsError, mean, pstdev
from typing import Dict, List, Mapping, Sequence, Tuple

from schemas.emotion import EmotionalState, EmotionType

from .state import BrainRuntimeConfig, PersonalityProfile


class EmotionModel:
    """Lightweight linear model mapping text/context features to VAD space."""

    HASH_BUCKETS = 16

    def __init__(
        self,
        weight_matrix: Sequence[Sequence[float]] | None = None,
        learning_rate: float = 0.03,
        regularization: float = 1e-3,
        momentum: float = 0.5,
    ) -> None:
        self.weight_matrix: List[List[float]] = [
            list(row)
            for row in (
                weight_matrix
                if weight_matrix is not None
                else self._default_weight_matrix()
            )
        ]
        self.learning_rate = max(1e-4, float(learning_rate))
        self.regularization = max(0.0, float(regularization))
        self.momentum = max(0.0, min(0.99, float(momentum)))
        self.weight_clip = 5.0
        self._velocity: List[List[float]] = [
            [0.0 for _ in row] for row in self.weight_matrix
        ]

    @staticmethod
    def _default_weight_matrix() -> List[List[float]]:
        """Return empirically tuned weights for valence/arousal/dominance."""

        return [
            [
                0.12,
                1.45,
                -0.08,
                0.05,
                0.90,
                0.35,
                0.25,
                -0.55,
                0.18,
                -0.10,
                0.12,
                0.50,
                -0.65,
                0.85,
                -1.25,
                0.40,
                0.30,
                0.35,
                -0.45,
                0.28,
                -0.50,
                0.35,
                0.14,
                -0.09,
                0.12,
                -0.11,
                0.08,
                -0.07,
                0.10,
                -0.06,
                0.09,
                -0.05,
                0.07,
                -0.04,
                0.06,
                -0.03,
                0.05,
                -0.02,
            ],
            [
                0.35,
                0.15,
                1.20,
                0.10,
                0.40,
                0.85,
                0.90,
                -0.25,
                0.12,
                0.35,
                0.08,
                0.12,
                0.25,
                -0.35,
                1.10,
                0.25,
                0.75,
                0.15,
                -0.65,
                0.10,
                -0.55,
                0.45,
                0.06,
                0.08,
                0.07,
                0.05,
                0.04,
                0.03,
                0.02,
                0.01,
                0.06,
                0.05,
                0.04,
                0.03,
                0.02,
                0.01,
                0.05,
                0.04,
            ],
            [
                0.05,
                0.10,
                -0.10,
                1.05,
                0.20,
                0.10,
                -0.05,
                -0.30,
                0.08,
                -0.05,
                0.03,
                0.25,
                -0.20,
                0.40,
                -0.60,
                0.25,
                0.10,
                0.60,
                -0.30,
                0.10,
                0.05,
                0.01,
                0.04,
                0.03,
                0.04,
                -0.02,
                0.03,
                -0.01,
                0.02,
                -0.01,
                0.03,
                -0.02,
                0.03,
                -0.02,
                0.02,
                -0.01,
                0.02,
                -0.01,
            ],
        ]

    def _hash_buckets(self, tokens: Sequence[str]) -> List[float]:
        buckets = [0.0] * self.HASH_BUCKETS
        if not tokens:
            return buckets
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = digest[0] % self.HASH_BUCKETS
            buckets[index] += 1.0
        total = sum(buckets)
        if total:
            buckets = [value / total for value in buckets]
        return buckets

    def feature_vector(
        self,
        lexical_vad: Dict[str, float],
        features: Dict[str, float],
        context: Dict[str, float],
        tokens: Sequence[str],
        positive_ratio: float,
        negative_ratio: float,
    ) -> List[float]:
        vector = [
            1.0,
            lexical_vad.get("valence", 0.0),
            lexical_vad.get("arousal", 0.0),
            lexical_vad.get("dominance", 0.0),
            features.get("sentiment", 0.0),
            features.get("intensity", 0.0),
            features.get("activation", 0.0),
            features.get("negation_density", 0.0),
            features.get("emphasis", 0.0),
            features.get("questions", 0.0),
            features.get("coverage", 0.0),
            positive_ratio,
            negative_ratio,
            context.get("safety", 0.0),
            context.get("threat", 0.0),
            context.get("social", 0.0),
            context.get("novelty", 0.0),
            context.get("control", 0.0),
            context.get("fatigue", 0.0),
            features.get("sentiment", 0.0) * context.get("social", 0.0),
            features.get("sentiment", 0.0) * context.get("threat", 0.0),
            features.get("intensity", 0.0) * context.get("novelty", 0.0),
        ]
        vector.extend(self._hash_buckets(tokens))
        return vector

    def predict(
        self,
        lexical_vad: Dict[str, float],
        features: Dict[str, float],
        context: Dict[str, float],
        tokens: Sequence[str],
        positive_ratio: float,
        negative_ratio: float,
    ) -> Tuple[Dict[str, float], List[float], List[float]]:
        vector = self.feature_vector(
            lexical_vad,
            features,
            context,
            tokens,
            positive_ratio,
            negative_ratio,
        )
        if any(len(row) != len(vector) for row in self.weight_matrix):
            raise ValueError("weight matrix shape does not match feature vector length")
        logits = [
            sum(weight * feature for weight, feature in zip(row, vector))
            for row in self.weight_matrix
        ]
        valence = math.tanh(logits[0])
        arousal = 1.0 / (1.0 + math.exp(-logits[1]))
        dominance = math.tanh(logits[2])
        dimensions = {
            "valence": max(-1.0, min(1.0, valence)),
            "arousal": max(0.0, min(1.0, arousal)),
            "dominance": max(-1.0, min(1.0, dominance)),
        }
        return dimensions, logits, vector

    def update(
        self,
        vector: Sequence[float],
        targets: Mapping[str, float],
        logits: Sequence[float],
    ) -> None:
        if not vector or len(logits) < 3:
            return
        if any(len(row) != len(vector) for row in self.weight_matrix):
            return
        outputs = [
            math.tanh(logits[0]),
            1.0 / (1.0 + math.exp(-logits[1])),
            math.tanh(logits[2]),
        ]
        derivatives = [
            1.0 - outputs[0] ** 2,
            outputs[1] * (1.0 - outputs[1]),
            1.0 - outputs[2] ** 2,
        ]
        for idx, key in enumerate(("valence", "arousal", "dominance")):
            derivative = derivatives[idx]
            if derivative == 0.0:
                continue
            target_value = float(targets.get(key, outputs[idx]))
            error = target_value - outputs[idx]
            gradient_scale = error * derivative
            if abs(gradient_scale) < 1e-9:
                continue
            row = self.weight_matrix[idx]
            velocity_row = self._velocity[idx]
            for j, feature in enumerate(vector):
                feature_value = float(feature)
                gradient = gradient_scale * feature_value - self.regularization * row[j]
                velocity_row[j] = (
                    self.momentum * velocity_row[j]
                    + self.learning_rate * gradient
                )
                row[j] += velocity_row[j]
                if row[j] > self.weight_clip:
                    row[j] = self.weight_clip
                elif row[j] < -self.weight_clip:
                    row[j] = -self.weight_clip


class EmotionProcessor:
    """Data-driven emotion classifier using lightweight VAD regression."""

    DEFAULT_VAD_LEXICON: Dict[str, Tuple[float, float, float]] = {
        "good": (0.72, 0.46, 0.54),
        "great": (0.85, 0.62, 0.60),
        "happy": (0.89, 0.72, 0.55),
        "joy": (0.95, 0.78, 0.55),
        "love": (0.93, 0.74, 0.52),
        "calm": (0.60, 0.20, 0.45),
        "relaxed": (0.70, 0.25, 0.55),
        "relief": (0.70, 0.32, 0.40),
        "secure": (0.68, 0.28, 0.58),
        "proud": (0.82, 0.50, 0.70),
        "win": (0.86, 0.65, 0.70),
        "success": (0.80, 0.58, 0.60),
        "bad": (-0.60, 0.50, -0.50),
        "sad": (-0.75, 0.45, -0.40),
        "angry": (-0.82, 0.78, 0.45),
        "terrible": (-0.85, 0.75, -0.60),
        "fail": (-0.74, 0.60, -0.55),
        "loss": (-0.70, 0.52, -0.50),
        "fear": (-0.60, 0.85, -0.60),
        "panic": (-0.70, 0.90, -0.70),
        "worry": (-0.55, 0.60, -0.55),
        "furious": (-0.80, 0.85, 0.30),
        "threat": (-0.70, 0.82, -0.60),
        "danger": (-0.70, 0.86, -0.55),
        "disgust": (-0.80, 0.68, -0.65),
        "grief": (-0.72, 0.50, -0.60),
        "excited": (0.88, 0.85, 0.60),
        "energized": (0.85, 0.90, 0.55),
        "surprise": (0.20, 0.75, -0.10),
        "bored": (-0.30, 0.20, -0.30),
        "resent": (-0.70, 0.60, -0.40),
        "hope": (0.78, 0.48, 0.55),
        "peace": (0.76, 0.25, 0.58),
        "support": (0.74, 0.32, 0.52),
    }
    INTENSIFIERS = {"very": 0.35, "extremely": 0.50, "incredibly": 0.45, "super": 0.20, "really": 0.20, "so": 0.15}
    NEGATIONS = {"not", "never", "no", "hardly", "barely"}
    ACTIVATION_TERMS = {
        "urgent",
        "surprise",
        "alert",
        "shock",
        "excited",
        "furious",
        "panic",
        "energized",
        "thrill",
        "alarm",
        "intense",
        "pressure",
    }

    def __init__(self, lexicon: Dict[str, Tuple[float, float, float]] | None = None) -> None:
        self.vad_lexicon = dict(lexicon or self.DEFAULT_VAD_LEXICON)
        self.baseline = {"valence": 0.0, "arousal": 0.35, "dominance": 0.05}
        self.positive_terms = {term for term, (valence, _, _) in self.vad_lexicon.items() if valence > 0.35}
        self.negative_terms = {term for term, (valence, _, _) in self.vad_lexicon.items() if valence < -0.35}
        self.model = EmotionModel()
        self.last_inference: Dict[str, List[float]] | None = None

    def _tokenize(self, stimulus: str) -> List[str]:
        return re.findall(r"[\w']+", stimulus.lower())

    def _aggregate_vad(self, tokens: List[str]) -> Dict[str, float]:
        totals = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        total_weight = 0.0
        negated = False
        pending_intensity = 0.0
        for token in tokens:
            if token in self.INTENSIFIERS:
                pending_intensity = self.INTENSIFIERS[token]
                continue
            if token in self.NEGATIONS:
                negated = not negated
                continue
            if token not in self.vad_lexicon:
                pending_intensity = 0.0
                continue
            weight = 1.0 + pending_intensity
            pending_intensity = 0.0
            valence, arousal, dominance = self.vad_lexicon[token]
            if negated:
                valence = -valence * 0.85
                dominance = -dominance * 0.65
                arousal = max(0.0, arousal * 0.75)
                negated = False
            totals["valence"] += valence * weight
            totals["arousal"] += arousal * weight
            totals["dominance"] += dominance * weight
            total_weight += weight
        if total_weight:
            return {k: v / total_weight for k, v in totals.items()}
        return dict(self.baseline)

    def _textual_features(self, tokens: List[str], stimulus: str) -> Dict[str, float]:
        positive_hits = sum(1 for token in tokens if token in self.positive_terms)
        negative_hits = sum(1 for token in tokens if token in self.negative_terms)
        sentiment_total = positive_hits + negative_hits
        lexical_sentiment = 0.0
        if sentiment_total:
            lexical_sentiment = (positive_hits - negative_hits) / sentiment_total
        activation_hits = sum(1 for token in tokens if token in self.ACTIVATION_TERMS)
        exclaim_count = stimulus.count("!")
        question_count = stimulus.count("?")
        emphasis_count = sum(
            1 for word in re.findall(r"\b\w+\b", stimulus) if word.isupper() and len(word) > 2
        )
        negation_hits = sum(1 for token in tokens if token in self.NEGATIONS)
        lexical_intensity = min(
            1.0, activation_hits * 0.25 + exclaim_count * 0.08 + emphasis_count * 0.12
        )
        token_count = max(1, len(tokens))
        exclaim_density = exclaim_count / token_count
        uppercase_density = emphasis_count / token_count
        try:
            mean_length = mean(len(token) for token in tokens) if tokens else 0.0
        except StatisticsError:
            mean_length = 0.0
        try:
            length_std = pstdev(len(token) for token in tokens) if len(tokens) > 1 else 0.0
        except StatisticsError:
            length_std = 0.0
        return {
            "sentiment": lexical_sentiment,
            "intensity": lexical_intensity,
            "activation": min(1.0, activation_hits * 0.2),
            "negation_density": negation_hits / max(1, len(tokens)),
            "emphasis": min(1.0, emphasis_count * 0.25),
            "questions": min(1.0, question_count * 0.25),
            "coverage": sentiment_total / max(1, len(tokens)),
            "exclaim_density": exclaim_density,
            "uppercase_density": uppercase_density,
            "mean_token_length": float(mean_length),
            "token_length_std": float(length_std),
            "positive_hits": float(positive_hits),
            "negative_hits": float(negative_hits),
            "token_count": float(token_count),
        }

    def _normalize_context(self, context: Dict[str, float] | None) -> Dict[str, float]:
        if not context:
            return {}
        normalized: Dict[str, float] = {}
        for key, value in context.items():
            try:
                normalized[key] = max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                continue
        return normalized

    def _derive_targets(
        self,
        lexical_vad: Dict[str, float],
        features: Dict[str, float],
        context: Dict[str, float],
    ) -> Dict[str, float]:
        sentiment = float(features.get("sentiment", 0.0))
        intensity = float(features.get("intensity", 0.0))
        activation = float(features.get("activation", 0.0))
        emphasis = float(features.get("emphasis", 0.0))
        safety = float(context.get("safety", 0.0))
        threat = float(context.get("threat", 0.0))
        social = float(context.get("social", 0.0))
        novelty = float(context.get("novelty", 0.0))
        control = float(context.get("control", 0.0))
        fatigue = float(context.get("fatigue", 0.0))
        base_valence = float(lexical_vad.get("valence", 0.0))
        base_arousal = float(lexical_vad.get("arousal", 0.5))
        base_dominance = float(lexical_vad.get("dominance", 0.0))
        target_valence = base_valence + sentiment * 0.4 + safety * 0.3 - threat * 0.45 + social * 0.2
        target_arousal = (
            base_arousal
            + intensity * 0.35
            + activation * 0.25
            + novelty * 0.3
            - fatigue * 0.4
        )
        target_dominance = (
            base_dominance
            + control * 0.5
            + safety * 0.2
            - threat * 0.25
            + emphasis * 0.2
        )
        return {
            "valence": max(-1.0, min(1.0, target_valence)),
            "arousal": max(0.0, min(1.0, target_arousal)),
            "dominance": max(-1.0, min(1.0, target_dominance)),
        }

    def evaluate(
        self, stimulus: str, context: Dict[str, float] | None = None
    ) -> Tuple[EmotionType, Dict[str, float], Dict[str, float]]:
        tokens = self._tokenize(stimulus)
        lexical_vad = self._aggregate_vad(tokens)
        features = self._textual_features(tokens, stimulus)
        context_weights = self._normalize_context(context)

        token_count = max(1, int(features.get("token_count", len(tokens)) or 1))
        positive_ratio = float(features.get("positive_hits", 0.0)) / token_count
        negative_ratio = float(features.get("negative_hits", 0.0)) / token_count

        dimensions, logits, vector = self.model.predict(
            lexical_vad,
            features,
            context_weights,
            tokens,
            positive_ratio,
            negative_ratio,
        )
        targets = self._derive_targets(lexical_vad, features, context_weights)
        self.model.update(vector, targets, logits)
        self.last_inference = {
            "logits": logits,
            "features": vector,
        }
        emotion = self._classify_emotion(dimensions, features, context_weights)
        context_weights.setdefault("model_activation", features.get("activation", 0.0))
        context_weights.setdefault("model_intensity", features.get("intensity", 0.0))
        context_weights.setdefault("model_coverage", features.get("coverage", 0.0))
        return emotion, dimensions, context_weights

    def _classify_emotion(
        self,
        dimensions: Dict[str, float],
        features: Dict[str, float],
        context: Dict[str, float],
    ) -> EmotionType:
        valence = dimensions.get("valence", 0.0)
        arousal = dimensions.get("arousal", 0.0)
        dominance = dimensions.get("dominance", 0.0)
        lexical_sentiment = features.get("sentiment", 0.0)
        intensity = features.get("intensity", 0.0)
        activation = features.get("activation", 0.0)
        coverage = features.get("coverage", 0.0)
        exclaim_density = features.get("exclaim_density", 0.0)

        threat = context.get("threat", 0.0)
        safety = context.get("safety", 0.0)
        novelty = context.get("novelty", 0.0)

        if valence >= 0.35:
            if arousal >= 0.35 or dominance >= 0.1 or lexical_sentiment > 0.2:
                return EmotionType.HAPPY
            if safety >= 0.6 and dominance > -0.2:
                return EmotionType.HAPPY
            if novelty >= 0.6 and intensity >= 0.3:
                return EmotionType.HAPPY
            return EmotionType.NEUTRAL

        if valence <= -0.35:
            if threat >= 0.45 or arousal >= 0.55 or activation >= 0.35 or exclaim_density > 0.2:
                return EmotionType.ANGRY
            return EmotionType.SAD

        if lexical_sentiment <= -0.25 and coverage > 0.2:
            return EmotionType.SAD

        if threat >= 0.6 and dominance <= 0.0 and arousal >= 0.45:
            return EmotionType.ANGRY

        if intensity < 0.2 and abs(valence) < 0.1 and arousal < 0.45:
            return EmotionType.NEUTRAL

        if lexical_sentiment >= 0.25 and (arousal >= 0.35 or novelty >= 0.4):
            return EmotionType.HAPPY

        return EmotionType.NEUTRAL


class MemoryConsolidator:
    """Store stimulus and emotion pairs for rudimentary memory."""

    def __init__(self, max_items: int = 64) -> None:
        self.max_items = max_items
        self.memory: List[Tuple[str, EmotionType, Dict[str, float]]] = []

    def consolidate(self, stimulus: str, emotion: EmotionType, dimensions: Dict[str, float]) -> None:
        self.memory.append((stimulus, emotion, dict(dimensions)))
        if len(self.memory) > self.max_items:
            self.memory.pop(0)

    def recent(self, limit: int = 5) -> List[Tuple[str, EmotionType, Dict[str, float]]]:
        return self.memory[-limit:]


class HomeostasisController:
    """Regulate the intensity of an :class:`EmotionalState` and track mood."""

    def __init__(self, decay_rate: float = 0.15, set_point: Dict[str, float] | None = None) -> None:
        self.mood: float = 0.0  # range [-1, 1]
        self.decay_rate = max(0.0, min(1.0, decay_rate))
        self.set_point = set_point or {"valence": 0.05, "arousal": 0.35, "dominance": 0.05}
        self.mood_inertia = 1.0 - self.decay_rate * 0.6
        self.last_dimensions: Dict[str, float] = dict(self.set_point)

    def regulate(
        self,
        state: EmotionalState,
        personality: PersonalityProfile,
        context: Dict[str, float],
        enable_decay: bool,
        full_dimensions: Dict[str, float] | None = None,
    ) -> EmotionalState:
        if enable_decay:
            self.mood *= self.mood_inertia
        else:
            self.mood *= 1.0 - self.decay_rate * 0.2

        dims_source = full_dimensions or state.dimensions
        valence = dims_source.get("valence", self.set_point["valence"])
        arousal = dims_source.get("arousal", state.intensity if "arousal" not in dims_source else dims_source["arousal"])
        dominance = dims_source.get("dominance", self.set_point["dominance"])
        self.last_dimensions = {
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
        }

        safety = context.get("safety", 0.0)
        threat = context.get("threat", 0.0)
        social = context.get("social", 0.0)
        novelty = context.get("novelty", 0.0)
        fatigue = context.get("fatigue", 0.0)

        valence_error = valence - self.set_point["valence"]
        arousal_error = arousal - self.set_point["arousal"]
        dominance_error = dominance - self.set_point["dominance"]

        mood_delta = valence_error * (0.48 + personality.extraversion * 0.30)
        mood_delta -= max(0.0, -valence_error) * (0.38 + personality.neuroticism * 0.35)
        mood_delta += dominance_error * 0.25
        mood_delta -= abs(arousal_error) * 0.10
        mood_delta += safety * 0.12 + social * 0.08
        mood_delta -= threat * (0.28 + personality.neuroticism * 0.20)
        self.mood = max(-1.0, min(1.0, self.mood + mood_delta))

        affective_drive = (
            abs(valence_error) * 0.45
            + max(0.0, arousal_error) * 0.35
            + abs(dominance_error) * 0.25
        )
        context_drive = (
            threat * (0.40 + personality.neuroticism * 0.30)
            + novelty * (0.30 + personality.openness * 0.20)
            + safety * (0.10 + personality.agreeableness * 0.10)
        )
        mood_drive = abs(self.mood) * 0.30
        dominance_drive = max(0.0, dominance_error) * 0.20

        intensity_target = 0.25 + affective_drive + context_drive + mood_drive + dominance_drive
        if valence < self.set_point["valence"]:
            intensity_target += personality.neuroticism * 0.20
        else:
            intensity_target += personality.extraversion * 0.15
        intensity_target -= fatigue * 0.20
        intensity_target = max(0.0, min(1.0, intensity_target))

        smoothing = 0.40 + self.decay_rate * 0.30 if enable_decay else 0.30
        smoothing = min(0.95, max(0.10, smoothing))
        state.intensity = max(0.0, min(1.0, state.intensity * (1 - smoothing) + intensity_target * smoothing))
        state.decay = self.decay_rate if enable_decay else 0.0
        return state


class LimbicSystem:
    """High level facade coordinating limbic sub-modules."""

    def __init__(self, personality: PersonalityProfile | None = None) -> None:
        self.emotion_processor = EmotionProcessor()
        self.memory_consolidator = MemoryConsolidator()
        self.homeostasis_controller = HomeostasisController()
        self.personality = personality or PersonalityProfile()
        self.personality.clamp()

    def build_state(
        self,
        emotion: EmotionType,
        full_dimensions: Dict[str, float],
        context_weights: Dict[str, float],
        config: BrainRuntimeConfig | None = None,
        stimulus: str | None = None,
    ) -> EmotionalState:
        """Construct an :class:`EmotionalState` from model outputs."""

        config = config or BrainRuntimeConfig()
        full_dimensions = dict(full_dimensions)
        context_weights = dict(context_weights)
        if config.enable_multi_dim_emotion:
            dimensions = dict(full_dimensions)
        else:
            dimensions = {"valence": full_dimensions.get("valence", 0.0)}

        valence = full_dimensions.get("valence", 0.0)
        arousal = full_dimensions.get("arousal", 0.35)
        dominance = full_dimensions.get("dominance", 0.05)
        base_intensity = 0.28 + abs(valence) * 0.32 + arousal * 0.30 + max(0.0, -dominance) * 0.08
        if emotion != EmotionType.NEUTRAL:
            base_intensity += 0.10
        base_intensity = max(0.0, min(1.0, base_intensity))
        if config.enable_personality_modulation:
            if valence >= 0:
                base_intensity += self.personality.extraversion * 0.10
            else:
                base_intensity += self.personality.neuroticism * 0.12
            base_intensity += (self.personality.openness - 0.5) * 0.06 * (arousal - 0.35)
            base_intensity += (self.personality.agreeableness - 0.5) * 0.04 * context_weights.get("social", 0.0)
            base_intensity = max(0.0, min(1.0, base_intensity))

        state = EmotionalState(
            emotion=emotion,
            intensity=base_intensity,
            dimensions=dimensions,
            context_weights=context_weights,
        )
        state = self.homeostasis_controller.regulate(
            state,
            self.personality if config.enable_personality_modulation else PersonalityProfile(),
            context_weights,
            enable_decay=config.enable_emotion_decay,
            full_dimensions=full_dimensions,
        )
        state.intent_bias = self._intent_bias(
            state, context_weights, config, full_dimensions=full_dimensions
        )
        if stimulus:
            self.memory_consolidator.consolidate(stimulus, emotion, full_dimensions)
        return state

    def react(
        self,
        stimulus: str,
        context: Dict[str, float] | None = None,
        config: BrainRuntimeConfig | None = None,
    ) -> EmotionalState:
        """Process ``stimulus`` and return the resulting emotional state."""

        config = config or BrainRuntimeConfig()
        emotion, full_dimensions, context_weights = self.emotion_processor.evaluate(stimulus, context)
        return self.build_state(
            emotion,
            full_dimensions,
            context_weights,
            config=config,
            stimulus=stimulus,
        )

    def _intent_bias(
        self,
        state: EmotionalState,
        context: Dict[str, float],
        config: BrainRuntimeConfig,
        full_dimensions: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        dims = full_dimensions or state.dimensions
        valence = dims.get("valence", state.dimensions.get("valence", 0.0))
        arousal = dims.get("arousal", state.dimensions.get("arousal", 0.0))
        dominance = dims.get("dominance", 0.0)
        novelty = context.get("novelty", 0.0)
        threat = context.get("threat", 0.0)
        safety = context.get("safety", 0.0)
        fatigue = context.get("fatigue", 0.0)

        approach_drive = max(0.0, 0.45 + 0.45 * valence + 0.20 * dominance + max(0.0, arousal - 0.4) * 0.20)
        withdraw_drive = max(
            0.0,
            0.45 - 0.35 * valence - 0.15 * dominance + threat * 0.50 + max(0.0, -dominance) * 0.20,
        )
        explore_drive = max(0.0, 0.30 + 0.55 * arousal + 0.30 * novelty - fatigue * 0.25)
        soothe_drive = max(0.0, 0.25 + 0.30 * (1 - arousal) + max(0.0, -valence) * 0.20 + safety * 0.20)

        bias = {
            "approach": approach_drive,
            "withdraw": withdraw_drive,
            "explore": explore_drive,
            "soothe": soothe_drive,
        }
        if config.enable_personality_modulation:
            bias["approach"] *= 0.55 + self.personality.extraversion * 0.45
            bias["withdraw"] *= 0.55 + self.personality.neuroticism * 0.45
            bias["explore"] *= 0.50 + self.personality.openness * 0.50
            bias["soothe"] *= 0.55 + (
                0.5 * self.personality.agreeableness + 0.5 * self.personality.conscientiousness
            )
        total = sum(bias.values()) or 1.0
        return {k: min(1.0, max(0.0, v / total)) for k, v in bias.items()}

    @property
    def mood(self) -> float:
        return self.homeostasis_controller.mood

    def update_personality(self, profile: PersonalityProfile) -> None:
        profile.clamp()
        self.personality = profile


__all__ = [
    "EmotionModel",
    "EmotionProcessor",
    "MemoryConsolidator",
    "HomeostasisController",
    "LimbicSystem",
]

