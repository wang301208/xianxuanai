"""Active learning samplers for prioritising informative samples.

The sampler ranks samples either by model uncertainty or diversity in feature
space. It is intended to be lightweight and independent from any specific
model implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class ActiveLearningSampler:
    """Select samples using uncertainty or diversity based heuristics."""

    strategy: str = "uncertainty"

    def score_uncertainty(self, probs: Sequence[Sequence[float]]) -> np.ndarray:
        """Return uncertainty scores given class probability distributions."""
        arr = np.asarray(probs)
        return 1.0 - arr.max(axis=1)

    def score_diversity(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        """Return diversity scores based on distance from the feature centroid."""
        feats = np.asarray(features)
        centroid = feats.mean(axis=0)
        return np.linalg.norm(feats - centroid, axis=1)

    def select(self, *, probs: Sequence[Sequence[float]] | None = None,
               features: Sequence[Sequence[float]] | None = None,
               k: int = 1) -> np.ndarray:
        """Return indices of the ``k`` most informative samples."""
        if self.strategy == "uncertainty" and probs is not None:
            scores = self.score_uncertainty(probs)
        elif features is not None:
            scores = self.score_diversity(features)
        else:
            raise ValueError("Insufficient data provided for sampling")
        idx = np.argsort(scores)[::-1][:k]
        return idx
