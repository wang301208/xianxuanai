"""General-purpose multimodal observation store.

This module augments :mod:`backend.world_model` with utilities for handling
arbitrary sensory modalities (audio, tactile, etc.).  Raw sensor readings can
be registered along with optional feature encoders.  Whenever new data arrives
the store keeps track of the latest reading per agent and fuses all available
modalities into a shared representation using the
:class:`~modules.brain.multimodal.MultimodalFusionEngine`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - torch not available in minimal envs
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - module may not be available in minimal tests
    from modules.brain.multimodal import MultimodalFusionEngine
except Exception:  # pragma: no cover - fallback ensures graceful degradation

    class MultimodalFusionEngine:  # type: ignore[override]
        """Minimal fallback that averages modality vectors when core module is absent."""

        def fuse_sensory_modalities(self, **modalities: np.ndarray) -> np.ndarray:
            if not modalities:
                raise ValueError("at least one modality is required")
            arrays = [np.asarray(array, dtype=float).reshape(-1) for array in modalities.values()]
            if len(arrays) == 1:
                return arrays[0]
            min_dim = min(array.shape[-1] for array in arrays)
            trimmed = [array[..., :min_dim] for array in arrays]
            stacked = np.stack(trimmed)
            return stacked.mean(axis=0)

logger = logging.getLogger(__name__)

SensorEncoder = Callable[[Any], Any]


def _to_numpy(value: Any) -> np.ndarray:
    """Best-effort conversion of ``value`` into a NumPy vector."""

    if value is None:
        return np.asarray([], dtype=float)

    if torch is not None and isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)

    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim > 1:
        return array.reshape(-1)
    return array.astype(float, copy=False)


def _sanitize_vector(value: Any) -> Optional[np.ndarray]:
    """Convert ``value`` into a non-empty 1D vector when feasible."""

    try:
        vector = _to_numpy(value)
    except Exception:  # pragma: no cover - defensive
        return None
    if vector.size == 0:
        return None
    return vector


def _fusable_modalities(modalities: Dict[str, "ModalityRecord"]) -> Dict[str, np.ndarray]:
    """Return features for modalities that currently have encodings."""

    fused: Dict[str, np.ndarray] = {}
    for name, record in modalities.items():
        if record.features is not None and record.features.size:
            fused[name] = record.features
    return fused


def _serialise_vector(vector: Optional[np.ndarray]) -> Optional[List[float]]:
    if vector is None:
        return None
    return vector.astype(float).tolist()


@dataclass
class ModalityRecord:
    """Store the latest raw observation and encoded features."""

    raw: Any | None = None
    features: Optional[np.ndarray] = None


@dataclass
class AgentRecord:
    """Container for all modalities recorded for a single agent."""

    modalities: Dict[str, ModalityRecord] = field(default_factory=dict)
    unified: Optional[np.ndarray] = None


class MultimodalStore:
    """Keep track of sensor observations across agents and modalities."""

    def __init__(self, fusion_engine: Optional[MultimodalFusionEngine] = None) -> None:
        self._fusion_engine = fusion_engine or MultimodalFusionEngine()
        self._encoders: Dict[str, SensorEncoder] = {}
        self._agents: Dict[str, AgentRecord] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_modality(self, name: str, encoder: Optional[SensorEncoder] = None) -> None:
        """Register ``encoder`` for ``name`` so raw samples can be vectorised."""

        if not name:
            raise ValueError("modality name must be a non-empty string")
        if encoder is not None and not callable(encoder):
            raise TypeError("encoder must be callable")
        self._encoders[name] = encoder or (lambda value: value)

    # ------------------------------------------------------------------
    # Observation ingestion
    # ------------------------------------------------------------------
    def ingest(
        self,
        agent_id: str,
        modality: str,
        *,
        data: Any | None = None,
        features: Any | None = None,
    ) -> Optional[np.ndarray]:
        """Record new observation for ``agent_id`` and update its unified state."""

        if not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        if not modality:
            raise ValueError("modality must be a non-empty string")

        agent = self._agents.setdefault(agent_id, AgentRecord())
        record = agent.modalities.setdefault(modality, ModalityRecord())

        if data is not None:
            record.raw = data

        vector: Optional[np.ndarray] = None
        if features is not None:
            vector = _sanitize_vector(features)
        elif modality in self._encoders and record.raw is not None:
            encoder = self._encoders[modality]
            try:
                encoded = encoder(record.raw)
                vector = _sanitize_vector(encoded)
            except Exception:  # pragma: no cover - encoder failures are non-fatal
                logger.warning("Encoder for modality '%s' failed", modality, exc_info=True)
        elif data is not None:
            vector = _sanitize_vector(data)

        if vector is not None:
            record.features = vector

        agent.modalities[modality] = record
        agent.unified = self._fuse(agent.modalities)
        self._agents[agent_id] = agent
        return agent.unified

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def unified(self, agent_id: str) -> Optional[np.ndarray]:
        """Return the current fused representation for ``agent_id``."""

        return self._agents.get(agent_id, AgentRecord()).unified

    def modalities(self, agent_id: str) -> Dict[str, ModalityRecord]:
        """Return the modality records for ``agent_id``."""

        return dict(self._agents.get(agent_id, AgentRecord()).modalities)

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Return a serialisable snapshot of all stored observations."""

        return {
            agent_id: {
                "unified": _serialise_vector(record.unified),
                "modalities": {
                    name: {
                        "raw": modality.raw,
                        "features": _serialise_vector(modality.features),
                    }
                    for name, modality in record.modalities.items()
                },
            }
            for agent_id, record in self._agents.items()
        }

    def agents(self) -> Iterable[str]:
        """Return identifiers for all agents with stored observations."""

        return tuple(self._agents.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fuse(self, modalities: Dict[str, ModalityRecord]) -> Optional[np.ndarray]:
        available = _fusable_modalities(modalities)
        if not available:
            return None
        if len(available) == 1:
            return next(iter(available.values()))
        try:
            return self._fusion_engine.fuse_sensory_modalities(**available)
        except Exception:  # pragma: no cover - fusion should not break ingestion
            logger.warning("Fusion engine failed; returning average vector", exc_info=True)
            vectors = [np.asarray(vec, dtype=float).reshape(-1) for vec in available.values()]
            min_dim = min(vec.shape[0] for vec in vectors)
            trimmed = [vec[:min_dim] for vec in vectors]
            stacked = np.stack(trimmed)
            return stacked.mean(axis=0)


__all__ = ["MultimodalStore", "SensorEncoder", "ModalityRecord", "AgentRecord"]
