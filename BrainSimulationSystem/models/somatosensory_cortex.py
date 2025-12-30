"""
Somatosensory cortex processing module.

Implements basic feature extraction for tactile, pressure, temperature,
and proprioceptive signals. Designed to work with simple numeric inputs
or sensor dictionaries without external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


@dataclass
class SomatosensoryConfig:
    """Configuration for somatosensory feature extraction."""

    normalize: bool = True
    feature_dim: int = 64
    history_decay: float = 0.8


class SomatosensoryCortex:
    """Aggregates sensor measurements into compact feature vectors."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = SomatosensoryConfig(**(config or {}))
        self._history: Dict[str, np.ndarray] = {}

    def process(self, sensor_data: Any) -> Dict[str, Any]:
        """
        Process sensor measurements.

        Args:
            sensor_data: The incoming tactile/pressure/temperature data. Accepts:
                - Dict[str, Sequence[float]]
                - Sequence[float]
                - Iterable of measurement dicts

        Returns:
            Dict with embedding and per-channel statistics.
        """

        flattened, stats = self._flatten(sensor_data)
        if flattened.size == 0:
            return {"embedding": np.zeros(self.config.feature_dim, dtype=np.float32), "statistics": stats}

        if self.config.normalize:
            norm = np.linalg.norm(flattened)
            if norm > 0:
                flattened = flattened / norm

        embedding = self._to_embedding(flattened)
        self._update_history(flattened)

        return {
            "embedding": embedding,
            "statistics": stats,
        }

    # ------------------------------------------------------------------ #
    def _flatten(self, sensor_data: Any) -> tuple[np.ndarray, Dict[str, Any]]:
        stats: Dict[str, Any] = {}
        if isinstance(sensor_data, Mapping):
            values = []
            for key, value in sensor_data.items():
                array = np.asarray(value, dtype=np.float32).flatten()
                if array.size == 0:
                    continue
                stats[key] = {
                    "mean": float(array.mean()),
                    "std": float(array.std()),
                    "min": float(array.min()),
                    "max": float(array.max()),
                }
                values.append(array)
            flattened = np.concatenate(values) if values else np.array([], dtype=np.float32)
            return flattened, stats

        array = np.asarray(list(sensor_data), dtype=np.float32).flatten()
        if array.size:
            stats["global"] = {
                "mean": float(array.mean()),
                "std": float(array.std()),
                "min": float(array.min()),
                "max": float(array.max()),
            }
        return array, stats

    def _to_embedding(self, flattened: np.ndarray) -> np.ndarray:
        """Project the flattened sensor vector into a fixed-size embedding."""

        target_dim = self.config.feature_dim
        if flattened.size == target_dim:
            return flattened.astype(np.float32, copy=False)

        if flattened.size > target_dim:
            step = flattened.size // target_dim
            embedded = flattened[: step * target_dim].reshape(target_dim, step).mean(axis=1)
        else:
            repeats = int(np.ceil(target_dim / flattened.size))
            tiled = np.tile(flattened, repeats)[:target_dim]
            embedded = tiled

        return embedded.astype(np.float32, copy=False)

    def _update_history(self, flattened: np.ndarray) -> None:
        decay = self.config.history_decay
        if not self._history:
            self._history["running_mean"] = flattened.copy()
            self._history["running_std"] = np.zeros_like(flattened)
            return

        mean = self._history["running_mean"]
        std = self._history["running_std"]

        mean *= decay
        mean += (1 - decay) * flattened

        std *= decay
        std += (1 - decay) * np.abs(flattened - mean)

        self._history["running_mean"] = mean
        self._history["running_std"] = std
