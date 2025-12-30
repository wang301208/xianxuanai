"""Engine for fusing multiple sensory modalities.

The :class:`MultimodalFusionEngine` provides a tiny abstraction around a
``CrossModalTransformer``.  It exposes a dependency injection interface so
that different transformer implementations can be supplied in tests or by
other parts of the system.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .cross_modal_transformer import CrossModalTransformer


class MultimodalFusionEngine:
    """Fuse data from arbitrary modalities into a shared representation using attention."""

    def __init__(self, transformer: Optional[CrossModalTransformer] = None) -> None:
        self._transformer = transformer or CrossModalTransformer()

    def set_transformer(self, transformer: CrossModalTransformer) -> None:
        """Inject a different :class:`CrossModalTransformer` instance."""

        self._transformer = transformer

    def fuse_sensory_modalities(self, **modalities: Any) -> np.ndarray:
        """Return a unified representation of the provided modalities.

        Parameters
        ----------
        **modalities:
            Named modality arrays (e.g. ``visual``, ``auditory``, ``tactile``,
            ``smell`` or ``text``).  At least one modality must be supplied.
            Each modality is aligned via the configured transformer and fused
            using an attention mechanism that weights modalities by the
            magnitude of their aligned representations as well as optional
            similarity/confidence hints supplied by upstream components such as
            :class:`~modules.perception.semantic_bridge.SemanticBridge`.
        """

        if not modalities:
            raise ValueError("at least one modality must be provided")

        aligned, hints = self._align_modalities(modalities)
        weights = self._attention(aligned, hints)
        return np.average(aligned, axis=0, weights=weights)

    def _align_modalities(
        self, modalities: Dict[str, Any]
    ) -> Tuple[Sequence[np.ndarray], Sequence[Dict[str, Any]]]:
        """Project modalities into the shared representation space."""

        aligned: list[np.ndarray] = []
        hints: list[Dict[str, Any]] = []
        for name, payload in modalities.items():
            vector, metadata = self._coerce_payload(payload)
            projected = self._transformer.project(vector, modality=name, metadata=metadata)
            aligned.append(projected)
            hints.append(metadata)
        return aligned, hints

    @staticmethod
    def _coerce_payload(payload: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        if isinstance(payload, np.ndarray):
            return payload, {}
        metadata: Dict[str, Any] = {}
        vector: Any = payload
        if isinstance(payload, dict):
            for key in ("embedding", "vector", "features"):
                if key in payload and payload[key] is not None:
                    vector = payload[key]
                    break
            candidate = payload.get("metadata")
            if isinstance(candidate, dict):
                metadata.update(candidate)
            for key in ("confidence", "similarity", "score", "weight"):
                if key in payload and key not in metadata:
                    metadata[key] = payload[key]
        vector_array = np.asarray(vector, dtype=float)
        if vector_array.ndim == 0:
            vector_array = vector_array.reshape(1)
        if vector_array.ndim > 1:
            vector_array = vector_array.reshape(-1)
        return vector_array.astype(float), metadata

    @staticmethod
    def _attention(aligned: Sequence[np.ndarray], hints: Sequence[Dict[str, Any]]) -> np.ndarray:
        """Compute attention weights for a sequence of aligned modalities."""

        scores = np.array([np.linalg.norm(a) for a in aligned], dtype=float)
        confidences = np.array([MultimodalFusionEngine._confidence_hint(h) for h in hints], dtype=float)
        scores *= confidences
        if np.allclose(scores, 0):
            return np.full(len(aligned), 1.0 / len(aligned))
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum()

    @staticmethod
    def _confidence_hint(metadata: Dict[str, Any]) -> float:
        values: list[float] = []
        for key in ("confidence", "similarity", "score", "weight"):
            raw = metadata.get(key)
            if raw is None:
                continue
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                seq: list[float] = []
                for item in raw:
                    try:
                        seq.append(float(item))
                    except (TypeError, ValueError):
                        continue
                if seq:
                    values.append(max(seq))
            else:
                try:
                    values.append(float(raw))
                except (TypeError, ValueError):
                    continue

        if not values:
            return 1.0
        hint = max(values)
        if hint <= 0:
            return 1.0
        return float(min(2.5, hint))


__all__ = ["MultimodalFusionEngine"]
