"""Basic dataclasses and enums describing emotional state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class EmotionType(Enum):
    """Enumeration of simple emotion categories."""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"


@dataclass
class EmotionalState:
    """Represents the current emotional appraisal of a stimulus."""

    emotion: EmotionType
    intensity: float = 0.0
    dimensions: Dict[str, float] = field(default_factory=dict)
    decay: float = 0.0
    context_weights: Dict[str, float] = field(default_factory=dict)
    intent_bias: Dict[str, float] = field(default_factory=dict)


__all__ = ["EmotionType", "EmotionalState"]

