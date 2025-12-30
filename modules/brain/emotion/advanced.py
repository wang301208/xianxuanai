import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class EmotionSpace:
    """Three-dimensional emotional space using valence, arousal and dominance."""

    valence: float
    arousal: float
    dominance: float


class MultimodalEmotionNet:
    """Encodes text, audio and visual stimuli and fuses them via attention."""

    def encode_text(self, text: str) -> np.ndarray:
        valence = len(text) / 100.0
        return np.array([valence, 0.0, 0.0], dtype=float)

    def encode_audio(self, audio: List[float]) -> np.ndarray:
        if not audio:
            return np.zeros(3)
        arousal = float(np.mean(audio))
        return np.array([0.0, arousal, 0.0], dtype=float)

    def encode_visual(self, visual: List[float]) -> np.ndarray:
        if not visual:
            return np.zeros(3)
        dominance = float(np.mean(visual))
        return np.array([0.0, 0.0, dominance], dtype=float)

    def forward(
        self,
        text: Optional[str] = None,
        audio: Optional[List[float]] = None,
        visual: Optional[List[float]] = None,
    ) -> np.ndarray:
        features = []
        if text is not None:
            features.append(self.encode_text(text))
        if audio is not None:
            features.append(self.encode_audio(audio))
        if visual is not None:
            features.append(self.encode_visual(visual))
        if not features:
            return np.zeros(3)
        features = np.stack(features)
        norms = np.linalg.norm(features, axis=1)
        weights = np.exp(norms)
        weights = weights / weights.sum()
        fused = (features * weights[:, None]).sum(axis=0)
        return fused


class EmotionalMemorySystem:
    """Stores past emotional states for later retrieval."""

    def __init__(self) -> None:
        self.log: List[EmotionSpace] = []

    def store(self, emotion: EmotionSpace) -> None:
        self.log.append(emotion)


class EmotionRegulationStrategies:
    """Applies a sequence of regulation strategies to an emotional state."""

    def __init__(
        self,
        strategies: Optional[List[Callable[[EmotionSpace], EmotionSpace]]] = None,
    ) -> None:
        self.strategies = strategies or []

    def apply(self, emotion: EmotionSpace) -> EmotionSpace:
        for strategy in self.strategies:
            emotion = strategy(emotion)
        return emotion


class AdvancedEmotionalSystem:
    """Evaluates stimuli to produce regulated emotions and stores them."""

    def __init__(
        self,
        regulation_strategies: Optional[List[Callable[[EmotionSpace], EmotionSpace]]] = None,
    ) -> None:
        self.network = MultimodalEmotionNet()
        self.memory = EmotionalMemorySystem()
        self.regulator = EmotionRegulationStrategies(regulation_strategies)

    def evaluate(self, stimulus: Dict[str, Any]) -> EmotionSpace:
        vector = self.network.forward(
            text=stimulus.get("text"),
            audio=stimulus.get("audio"),
            visual=stimulus.get("visual"),
        )
        emotion = EmotionSpace(*vector)
        self.memory.store(emotion)
        regulated = self.regulator.apply(emotion)
        return regulated
