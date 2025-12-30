"""High-level perception modules bridging raw sensory data to cortex models."""

from .vision import VisionPerceptionModule, VisionPerceptionConfig
from .self_supervised import ContrastiveLearner, ContrastiveLearnerConfig

__all__ = [
    "VisionPerceptionModule",
    "VisionPerceptionConfig",
    "ContrastiveLearner",
    "ContrastiveLearnerConfig",
]
