from __future__ import annotations

"""Hierarchical consciousness model with emotional and metacognitive control.

The model organises information into per-module *local* workspaces and a
shared *global* workspace.  Attention thresholds are tracked for each module
and can adapt through metacognitive feedback or emotional/motivational
signals.  External observers may register callbacks to inspect or intervene in
runtime state.
"""

from collections import defaultdict
from typing import Any, Callable, Dict, List, DefaultDict


class HierarchicalConsciousnessModel:
    """Cognitive toy model supporting attention, emotion and feedback."""

    def __init__(self, base_threshold: float = 0.5) -> None:
        self.global_workspace: List[Dict[str, Any]] = []
        self.local_workspaces: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.thresholds: DefaultDict[str, float] = defaultdict(lambda: base_threshold)
        self.observers: List[Callable[[Dict[str, Any]], None]] = []
        self._predictions: List[bool] = []
        self._truths: List[bool] = []
        self.motivation: float = 0.0

    # ------------------------------------------------------------------
    # Interfaces
    # ------------------------------------------------------------------
    def register_observer(self, hook: Callable[[Dict[str, Any]], None]) -> None:
        """Allow external components to observe internal state changes."""

        self.observers.append(hook)

    def _notify(self) -> None:
        if not self.observers:
            return
        snapshot = {
            "global": list(self.global_workspace),
            "local": {k: list(v) for k, v in self.local_workspaces.items()},
            "thresholds": dict(self.thresholds),
            "motivation": self.motivation,
        }
        for hook in self.observers:
            hook(snapshot)

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------
    def focus_attention(self, module: str, information: Dict[str, Any]) -> bool:
        """Process ``information`` for ``module`` and broadcast if salient."""

        score = float(information.get("score", 0.0))
        threshold = self.thresholds[module]
        is_salient = score > threshold
        if is_salient:
            self.global_workspace.append(dict(information))
        else:
            self.local_workspaces[module].append(dict(information))

        if "ground_truth" in information:
            truth = bool(information["ground_truth"])
            self.metacognitive_feedback(module, is_salient, truth)

        self._notify()
        return is_salient

    # ------------------------------------------------------------------
    # Feedback and modulation
    # ------------------------------------------------------------------
    def metacognitive_feedback(self, module: str, prediction: bool, truth: bool) -> None:
        """Record performance and adapt module threshold."""

        self._predictions.append(prediction)
        self._truths.append(truth)
        if prediction and not truth:
            self.thresholds[module] = min(1.0, self.thresholds[module] + 0.1)
        elif not prediction and truth:
            self.thresholds[module] = max(0.0, self.thresholds[module] - 0.1)

    def input_emotional_signal(self, *, valence: float = 0.0, arousal: float = 0.0, motivation: float = 0.0) -> None:
        """Adjust thresholds based on emotional/motivational cues."""

        delta = 0.01 * (arousal - valence)
        self.motivation += motivation
        for module in list(self.thresholds.keys()):
            self.thresholds[module] = min(
                1.0,
                max(0.0, self.thresholds[module] + delta - 0.01 * motivation),
            )
        self._notify()

    def intervene(self, module: str, *, threshold: float | None = None) -> None:
        """External override for module parameters."""

        if threshold is not None:
            self.thresholds[module] = threshold
            self._notify()

    def recover_anomaly(self, module: str) -> None:
        """Escalate last local entry of ``module`` to the global workspace."""

        if self.local_workspaces[module]:
            info = self.local_workspaces[module].pop()
            self.global_workspace.append(info)
            self._notify()

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def accuracy(self) -> float:
        if not self._truths:
            return 0.0
        correct = sum(p == t for p, t in zip(self._predictions, self._truths))
        return correct / len(self._truths)

    def local_workspace(self, module: str) -> List[Dict[str, Any]]:
        return self.local_workspaces[module]

    def global_broadcasts(self) -> List[Dict[str, Any]]:
        return self.global_workspace


__all__ = ["HierarchicalConsciousnessModel"]
