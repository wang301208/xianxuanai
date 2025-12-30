"""Minimal quantum-inspired machine learning utilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

import numpy as np


@dataclass
class QuantumClassifier:
    """Very small quantum-style classifier.

    The classifier encodes feature vectors as quantum states by normalising the
    input vector.  For each class label the mean state is stored.  Prediction is
    performed by selecting the class whose mean state has the highest squared
    inner product with the encoded sample.
    """

    class_states: Dict[int, np.ndarray] = field(default_factory=dict)

    def _encode(self, x: Iterable[float]) -> np.ndarray:
        vec = np.array(list(x), dtype=float)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return np.array([1.0] + [0.0] * (len(vec) - 1))
        return vec / norm

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train classifier on feature matrix ``X`` and label vector ``y``."""

        for label in np.unique(y):
            samples = X[y == label]
            encoded = [self._encode(s) for s in samples]
            self.class_states[int(label)] = np.mean(encoded, axis=0)

    def predict(self, sample: Iterable[float]) -> int:
        """Return the predicted label for ``sample``."""

        vec = self._encode(sample)
        scores = {
            label: float(abs(np.vdot(vec, state)) ** 2)
            for label, state in self.class_states.items()
        }
        return max(scores, key=scores.get)


__all__ = ["QuantumClassifier"]
