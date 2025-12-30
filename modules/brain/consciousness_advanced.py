from __future__ import annotations

"""Advanced consciousness model with hierarchical workspace and plug-in strategies.

This module implements a more feature complete consciousness architecture than
``modules.brain.consciousness``.  It provides a hierarchical Global Workspace,
pluggable attention and memory strategies, a simple metacognitive monitor and an
emotion regulation hook.  A tiny ``visualize_runtime`` helper exposes the
internal attention scores for external tooling or interactive inspection.

The implementation intentionally remains lightweight – the goal of the file is
mainly to offer richer structure for unit tests rather than neuroscientific
fidelity.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Callable
from collections import defaultdict

# ---------------------------------------------------------------------------
#  Pluggable attention and memory strategies
# ---------------------------------------------------------------------------


class AttentionStrategy(Protocol):
    """Strategy interface returning a salience score for information."""

    def score(self, information: Dict[str, Any]) -> float:  # pragma: no cover - protocol
        ...


class ThresholdAttention:
    """Basic attention using a fixed threshold."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def score(self, information: Dict[str, Any]) -> float:
        return float(information.get("score", 0.0))

    def is_salient(self, information: Dict[str, Any]) -> bool:
        return self.score(information) > self.threshold


class AdaptiveAttention(ThresholdAttention):
    """Attention that adapts its threshold based on feedback."""

    def adapt(self, prediction: bool, truth: bool) -> None:
        if prediction and not truth:
            self.threshold = min(1.0, self.threshold + 0.05)
        elif not prediction and truth:
            self.threshold = max(0.0, self.threshold - 0.05)


class MemoryStrategy(Protocol):
    """Strategy interface for storing and retrieving information."""

    def store(self, information: Dict[str, Any]) -> None:  # pragma: no cover - protocol
        ...

    @property
    def data(self) -> List[Dict[str, Any]]:  # pragma: no cover - protocol
        ...


class ListMemory:
    """Simple in-memory log of processed information."""

    def __init__(self) -> None:
        self._data: List[Dict[str, Any]] = []

    def store(self, information: Dict[str, Any]) -> None:
        self._data.append(dict(information))

    @property
    def data(self) -> List[Dict[str, Any]]:
        return self._data


# ---------------------------------------------------------------------------
#  Hierarchical global workspace and metacognition
# ---------------------------------------------------------------------------


class HierarchicalGlobalWorkspace:
    """Maintain per-module local workspaces and a shared global workspace."""

    def __init__(self) -> None:
        self.local: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.global_broadcasts: List[Dict[str, Any]] = []

    def broadcast(self, module: str, information: Dict[str, Any], *, level: str = "local") -> None:
        if level == "global":
            self.global_broadcasts.append(dict(information))
        else:
            self.local[module].append(dict(information))

    def escalate(self, module: str) -> None:
        """Move the last local broadcast of ``module`` to the global workspace."""
        if self.local[module]:
            info = self.local[module][-1]
            self.global_broadcasts.append(info)


class Metacognition:
    """Track prediction accuracy and expose adaptive feedback."""

    def __init__(self) -> None:
        self._pred: List[bool] = []
        self._truth: List[bool] = []

    def record(self, prediction: bool, truth: bool, attention: AdaptiveAttention | None = None) -> None:
        self._pred.append(bool(prediction))
        self._truth.append(bool(truth))
        if attention is not None:
            attention.adapt(prediction, truth)

    def accuracy(self) -> float:
        if not self._truth:
            return 0.0
        correct = sum(p == t for p, t in zip(self._pred, self._truth))
        return correct / len(self._truth)


# ---------------------------------------------------------------------------
#  Emotion regulation wrapper
# ---------------------------------------------------------------------------

try:  # pragma: no cover - optional dependency
    from .emotion.advanced import AdvancedEmotionalSystem, EmotionSpace
except Exception:  # pragma: no cover
    AdvancedEmotionalSystem = None  # type: ignore
    EmotionSpace = None  # type: ignore


class EmotionRegulator:
    """Adjust attention threshold based on evaluated emotion."""

    def __init__(self) -> None:
        self.system = AdvancedEmotionalSystem() if AdvancedEmotionalSystem else None

    def regulate(self, attention: ThresholdAttention, stimulus: Dict[str, Any]) -> None:
        if not self.system:
            return
        emotion = self.system.evaluate(stimulus)
        # Simple rule: higher arousal lowers threshold slightly, negative valence raises it
        delta = 0.01 * (emotion.arousal - emotion.valence)
        attention.threshold = min(1.0, max(0.0, attention.threshold + delta))


# ---------------------------------------------------------------------------
#  Advanced consciousness model
# ---------------------------------------------------------------------------


@dataclass
class ConsciousnessAdvanced:
    """Integrate workspace, attention, memory, metacognition and emotion.

    The class exposes lightweight helpers for querying local/global workspaces,
    inspecting metacognitive accuracy and registering visualisation hooks that
    receive real‑time updates of attention and memory changes.
    """

    workspace: HierarchicalGlobalWorkspace = field(default_factory=HierarchicalGlobalWorkspace)
    attention: ThresholdAttention = field(default_factory=AdaptiveAttention)
    memory: MemoryStrategy = field(default_factory=ListMemory)
    meta: Metacognition = field(default_factory=Metacognition)
    emotion: EmotionRegulator = field(default_factory=EmotionRegulator)
    _visualizers: List[Callable[[Dict[str, Any]], None]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Interfaces for external components
    # ------------------------------------------------------------------
    def add_visualizer(self, hook: Callable[[Dict[str, Any]], None]) -> None:
        """Register a callback receiving runtime snapshots.

        ``hook`` should accept a single ``dict`` argument.  Each invocation
        contains the current attention score distribution, a copy of the working
        memory and the state of the hierarchical workspaces.
        """

        self._visualizers.append(hook)

    def global_workspace(self) -> List[Dict[str, Any]]:
        return self.workspace.global_broadcasts

    def local_workspace(self, module: str) -> List[Dict[str, Any]]:
        return self.workspace.local.get(module, [])

    def metacognitive_accuracy(self) -> float:
        return self.meta.accuracy()

    def regulate_emotions(self, stimulus: Dict[str, Any]) -> None:
        self.emotion.regulate(self.attention, stimulus)

    def _notify_visualizers(self, information: Dict[str, Any]) -> None:
        if not self._visualizers:
            return
        snapshot = {
            "attention_scores": [self.attention.score(item) for item in self.memory.data],
            "memory": list(self.memory.data),
            "global_workspace": list(self.workspace.global_broadcasts),
            "local_workspace": {k: list(v) for k, v in self.workspace.local.items()},
        }
        for hook in self._visualizers:
            hook(snapshot)

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------
    def conscious_access(self, information: Dict[str, Any], stimulus: Dict[str, Any] | None = None) -> bool:
        """Process information and broadcast if salient.

        ``information`` may include a numeric ``"score"`` and an optional
        ``"ground_truth"`` label.  ``stimulus`` is forwarded to the emotion
        regulator.
        """

        if stimulus:
            self.emotion.regulate(self.attention, stimulus)

        is_salient = False
        if hasattr(self.attention, "is_salient"):
            is_salient = bool(self.attention.is_salient(information))  # type: ignore[attr-defined]
        else:
            is_salient = self.attention.score(information) > 0.5

        if is_salient:
            self.workspace.broadcast("module", information, level="global")
            self.memory.store(information)
        else:
            self.workspace.broadcast("module", information, level="local")

        if "ground_truth" in information:
            truth = bool(information["ground_truth"])
            self.meta.record(is_salient, truth, self.attention if isinstance(self.attention, AdaptiveAttention) else None)

        self._notify_visualizers(information)
        return is_salient

    # Convenience helpers -------------------------------------------------

    def evaluate_dataset(self, dataset: List[Dict[str, Any]]) -> float:
        for item in dataset:
            self.conscious_access(item, stimulus={"text": item.get("text", "")})
        return self.meta.accuracy()

    def visualize_runtime(self):  # pragma: no cover - plotting utility
        import matplotlib.pyplot as plt

        scores = [self.attention.score(item) for item in self.memory.data]
        plt.figure()
        plt.plot(scores, marker="o")
        plt.title("Attention scores over time")
        plt.xlabel("Item")
        plt.ylabel("Score")
        plt.tight_layout()
        return plt


# Backwards compatibility alias
ConsciousnessAdvancedModel = ConsciousnessAdvanced


__all__ = [
    "AttentionStrategy",
    "ThresholdAttention",
    "AdaptiveAttention",
    "MemoryStrategy",
    "ListMemory",
    "HierarchicalGlobalWorkspace",
    "Metacognition",
    "EmotionRegulator",
    "ConsciousnessAdvanced",
    "ConsciousnessAdvancedModel",
]
