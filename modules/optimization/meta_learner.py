from __future__ import annotations

"""Simple meta-learning utilities for modality weight optimization."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MetaLearner:
    """Maintain and update weights for different modality generators."""

    weights: Dict[str, float] = field(default_factory=dict)

    def __init__(self, modalities: List[str]):
        self.weights = {m: 0.5 for m in modalities}

    def update(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Update modality weights towards observed ``scores``.

        The update rule performs a small step toward the provided score for each
        modality, enabling gradual self-optimization.
        """
        for modality, score in scores.items():
            w = self.weights.get(modality, 0.5)
            self.weights[modality] = w + 0.1 * (score - w)
        return self.weights


__all__ = ["MetaLearner"]
