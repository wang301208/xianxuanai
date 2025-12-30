from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from schemas.emotion import EmotionType
from .emotion_service import EmotionServiceConfig


@dataclass
class PersonalityProfile:
    """Simple personality trait container (0.0 - 1.0 range)."""

    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5

    def clamp(self) -> None:
        for attr in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
            value = getattr(self, attr)
            setattr(self, attr, max(0.0, min(1.0, value)))

    def modulation_weight(self, channel: str) -> float:
        """Return a weight for a downstream channel based on personality."""
        lookup = {
            "explore": self.openness,
            "persist": self.conscientiousness,
            "social": self.extraversion,
            "empathy": self.agreeableness,
            "caution": self.neuroticism,
        }
        return max(0.0, min(1.0, lookup.get(channel, 0.5)))

    @property
    def traits(self) -> Dict[str, float]:
        """Expose a mapping-style view of trait values for reinforcement policies."""

        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
        }


@dataclass
class BrainRuntimeConfig:
    """Runtime toggles controlling advanced brain features."""

    use_neuromorphic: bool = True
    enable_multi_dim_emotion: bool = True
    enable_emotion_decay: bool = True
    enable_curiosity_feedback: bool = True
    enable_self_learning: bool = True
    enable_continual_learning: bool = False
    enable_self_evolution: bool = False
    enable_meta_cognition: bool = False
    enable_personality_modulation: bool = True
    enable_plan_logging: bool = True
    prefer_structured_planner: bool = True
    prefer_reinforcement_planner: bool = False
    metrics_enabled: bool = True
    continual_learning_experience_root: str = "data/experience"
    continual_learning_background_interval: float = 60.0
    continual_learning_policy_path: str = "data/experience/policies/intention_bandit.json"
    evolution_ga_population: int = 20
    evolution_ga_generations: int = 5
    evolution_ga_mutation_rate: float = 0.3
    evolution_ga_mutation_sigma: float = 0.1
    meta_failure_threshold: int = 3
    meta_low_confidence_threshold: float = 0.35
    meta_low_confidence_window: int = 5
    meta_knowledge_gap_threshold: int = 2
    capability_prior: Dict[str, float] = field(default_factory=dict)
    hardware_targets: Dict[str, str] = field(default_factory=dict)
    emotion_service: Optional[EmotionServiceConfig] = None


@dataclass
class CuriosityState:
    """Track curiosity drive and novelty handling."""

    drive: float = 0.4
    novelty_preference: float = 0.5
    fatigue: float = 0.1
    last_novelty: float = 0.0

    def update(self, novelty: float, personality: PersonalityProfile) -> None:
        novelty = max(0.0, min(1.0, novelty))
        openness_bias = 0.3 + personality.openness * 0.7
        self.drive = max(0.0, min(1.0, self.drive * (1 - self.fatigue) + novelty * openness_bias))
        self.fatigue = max(0.0, min(1.0, self.fatigue * 0.8 + novelty * 0.1))
        self.last_novelty = novelty

    def decay(self) -> None:
        self.drive = max(0.0, self.drive * 0.95)
        self.fatigue = max(0.0, self.fatigue * 0.9)

    def as_metrics(self) -> Dict[str, float]:
        return {
            "curiosity_drive": self.drive,
            "curiosity_fatigue": self.fatigue,
            "novelty_preference": self.novelty_preference,
            "last_novelty": self.last_novelty,
        }


@dataclass
class PerceptionSnapshot:
    modalities: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    semantic: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    knowledge_facts: List[Dict[str, Any]] = field(default_factory=list)
    fused_embedding: List[float] = field(default_factory=list)
    modality_embeddings: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class EmotionSnapshot:
    primary: EmotionType
    intensity: float
    mood: float
    dimensions: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, float] = field(default_factory=dict)
    decay: float = 0.0
    intent_bias: Dict[str, float] = field(default_factory=dict)

    def as_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            "emotion_intensity": self.intensity,
            "emotion_mood": self.mood,
        }
        metrics.update({f"emotion_{k}": v for k, v in self.dimensions.items()})
        metrics.update({f"emotion_context_{k}": v for k, v in self.context.items()})
        metrics["emotion_decay"] = self.decay
        return metrics


@dataclass
class CognitiveIntent:
    intention: str
    salience: bool
    plan: List[str] = field(default_factory=list)
    confidence: float = 0.5
    weights: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def as_metrics(self) -> Dict[str, float]:
        metrics = {"intent_confidence": self.confidence}
        metrics.update({f"intent_weight_{k}": v for k, v in self.weights.items()})
        return metrics


@dataclass
class ThoughtSnapshot:
    """Representation of the agent's current thought focus."""

    focus: str
    summary: str
    plan: List[str] = field(default_factory=list)
    confidence: float = 0.5
    memory_refs: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def as_metrics(self) -> Dict[str, float]:
        return {
            "thought_confidence": self.confidence,
            "thought_plan_length": float(len(self.plan)),
        }


@dataclass
class FeelingSnapshot:
    """Subjective feeling derived from emotion dynamics."""

    descriptor: str
    valence: float
    arousal: float
    mood: float
    confidence: float
    context_tags: List[str] = field(default_factory=list)

    def as_metrics(self) -> Dict[str, float]:
        return {
            "feeling_valence": self.valence,
            "feeling_arousal": self.arousal,
            "feeling_confidence": self.confidence,
        }


@dataclass
class BrainCycleResult:
    perception: PerceptionSnapshot
    emotion: EmotionSnapshot
    intent: CognitiveIntent
    personality: PersonalityProfile
    curiosity: CuriosityState
    energy_used: int
    idle_skipped: int
    thoughts: Optional[ThoughtSnapshot] = None
    feeling: Optional[FeelingSnapshot] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Optional[str]] = field(default_factory=dict)


__all__ = [
    "PersonalityProfile",
    "BrainRuntimeConfig",
    "CuriosityState",
    "PerceptionSnapshot",
    "EmotionSnapshot",
    "CognitiveIntent",
    "ThoughtSnapshot",
    "FeelingSnapshot",
    "BrainCycleResult",
    "EmotionServiceConfig",
]


