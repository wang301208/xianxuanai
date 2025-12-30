"""Neuromorphic-quantum hybrid processing module."""
from __future__ import annotations

from typing import Iterable, List, Optional
import numpy as np

from modules.brain.neuromorphic import temporal_encoding


class NeuromorphicQuantumHybrid:
    """Combine neuromorphic preprocessing with quantum feature mapping."""

    def neuromorphic_preprocess(self, signal: Iterable[float]) -> List[int]:
        """Simple neuromorphic feature extraction via thresholding and latency encoding."""
        values = [1.0 if x >= 0.5 else 0.0 for x in signal]
        events = temporal_encoding.latency_encode(values)
        features = [0] * len(values)
        for _, spikes in events:
            for i, s in enumerate(spikes):
                features[i] += s
        return features

    def quantum_feature_map(self, features: Iterable[float]) -> np.ndarray:
        """Map classical features to a normalised quantum-style state vector."""
        vec = np.array(list(features), dtype=float)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def mixed_optimize(
        self,
        state: np.ndarray,
        target: np.ndarray,
        iterations: int = 10,
        lr: float = 0.1,
    ) -> np.ndarray:
        """Toy gradient-descent optimisation of ``state`` toward ``target``."""
        s = state.copy()
        for _ in range(iterations):
            grad = target - s
            s = s + lr * grad
            n = np.linalg.norm(s)
            if n != 0:
                s = s / n
        return s

    def hybrid_processing(
        self,
        signal: Iterable[float],
        *,
        target_state: Optional[Iterable[float]] = None,
        iterations: int = 10,
        lr: float = 0.1,
    ) -> np.ndarray:
        """Run neuromorphic preprocessing, quantum mapping and optional optimisation."""
        features = self.neuromorphic_preprocess(signal)
        state = self.quantum_feature_map(features)
        if target_state is not None:
            target = np.array(list(target_state), dtype=float)
            n = np.linalg.norm(target)
            if n != 0:
                target = target / n
            state = self.mixed_optimize(state, target, iterations, lr)
        return state


__all__ = ["NeuromorphicQuantumHybrid"]
