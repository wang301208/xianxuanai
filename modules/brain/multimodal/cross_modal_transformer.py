"""Learned projections for aligning heterogeneous sensory modalities.

The original implementation exposed a deliberately tiny stand-in that simply
repeated per-modality means to obtain a shared representation.  While
convenient for unit tests, that behaviour failed to capture any notion of a
shared semantic space.  The updated transformer introduces lightweight,
deterministic projection heads that mimic learned adapters.  When CLIP is
available we seed the projections for vision/text using its embeddings; audio
modalities receive a tiny tanh adapter; all other modalities fall back to a
randomly-initialised linear map.  The result is a richer alignment mechanism
that still keeps dependencies optional and fast for tests.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency when CLIP stack present
    from backend.ml.feature_extractor import CLIPFeatureExtractor
except Exception:  # pragma: no cover - keep transformer usable in minimal envs
    CLIPFeatureExtractor = None  # type: ignore[assignment]


def _to_numpy(value: Any) -> np.ndarray:
    """Best-effort conversion into a 1D numpy array."""

    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim > 1:
        array = array.reshape(-1)
    return array.astype(float, copy=False)


def _normalise(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector


class CrossModalTransformer:
    """Align and fuse multiple sensory modalities into one vector.

    Parameters
    ----------
    output_dim:
        Dimensionality of the shared representation space.  All modalities
        are projected to this size before fusion.
    """

    def __init__(self, output_dim: int = 16) -> None:
        self.output_dim = int(output_dim)
        self._projection_heads: Dict[Tuple[str, int], np.ndarray] = {}
        self._audio_heads: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._clip_anchors = self._initialise_clip_anchors()

    # ------------------------------------------------------------------
    # Projection interface
    # ------------------------------------------------------------------
    def project(
        self,
        x: np.ndarray,
        *,
        modality: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Public wrapper around :meth:`_project` for external callers.

        Exposing the projection step allows higher level modules to perform
        more sophisticated fusion strategies (e.g. attention mechanisms)
        while reusing the alignment logic implemented by this transformer.
        """

        return self._project(x, modality=modality, metadata=metadata or {})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _project(
        self,
        x: np.ndarray,
        *,
        modality: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Project an input modality to the shared representation space.

        The adapter uses modality-aware projection heads.  Vision/text inputs
        are guided by CLIP anchors when available, audio is passed through a
        tiny tanh adapter, and other modalities receive a deterministic linear
        projection.  All projections are normalised to the target dimensionality.
        """

        arr = _to_numpy(x)
        if arr.size == 0:
            raise ValueError("modality input is empty")

        modality_name = self._canonical_modality(modality)
        metadata = metadata or {}

        if modality_name in {"vision", "text"} and self._clip_anchors:
            return self._clip_projection(arr, modality_name, metadata)
        if modality_name == "audio":
            return self._audio_projection(arr)

        return self._linear_projection(arr, modality_name)

    def fuse(self, modalities: Sequence[np.ndarray]) -> np.ndarray:
        """Fuse multiple modalities into a unified representation.

        Parameters
        ----------
        modalities:
            Sequence of modality arrays to be fused.  Each modality is first
            aligned to the shared space and then averaged.

        Returns
        -------
        np.ndarray
            The fused representation of shape ``(output_dim,)``.
        """

        if not modalities or any(m is None for m in modalities):
            raise ValueError("modalities must be provided")
        aligned = [self._project(m) for m in modalities]
        return np.mean(aligned, axis=0)

    # ------------------------------------------------------------------
    # Projection heads
    # ------------------------------------------------------------------
    def _clip_projection(
        self,
        arr: np.ndarray,
        modality: str,
        metadata: Dict[str, Any],
    ) -> np.ndarray:
        anchor = self._clip_anchors.get(modality)
        if anchor is None:
            return self._linear_projection(arr, modality)

        arr = _normalise(arr)
        resized = self._resize(arr, self.output_dim)
        similarity_hint = self._extract_similarity(metadata)
        similarity = float(np.dot(resized, anchor))
        weight = max(0.1, min(2.0, similarity + similarity_hint + 1.0))
        blended = _normalise(resized * (2.0 - weight) + anchor * weight)
        return blended

    def _audio_projection(self, arr: np.ndarray) -> np.ndarray:
        weights, bias = self._audio_heads.setdefault(
            arr.size,
            self._initialise_audio_head(arr.size),
        )
        projected = np.tanh(arr @ weights + bias)
        return self._resize(_normalise(projected), self.output_dim)

    def _linear_projection(self, arr: np.ndarray, modality: str) -> np.ndarray:
        head = self._projection_heads.setdefault(
            (modality, arr.size),
            self._initialise_linear_head(modality, arr.size),
        )
        projected = arr @ head
        return self._resize(_normalise(projected), self.output_dim)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _initialise_clip_anchors(self) -> Dict[str, np.ndarray]:
        anchors: Dict[str, np.ndarray] = {}
        prompts = {
            "vision": "a vivid photo depicting an environment",
            "text": "a descriptive natural language summary",
        }
        extractor: Any | None = None
        if CLIPFeatureExtractor is not None:
            try:
                extractor = CLIPFeatureExtractor()
            except Exception:  # pragma: no cover - optional dependency failed
                extractor = None

        for modality, prompt in prompts.items():
            vector: Optional[np.ndarray] = None
            if extractor is not None:
                try:
                    tensor = extractor.extract_text_features(prompt)
                except Exception:  # pragma: no cover
                    tensor = None
                if tensor is not None:
                    try:
                        vector = _to_numpy(tensor)
                    except Exception:  # pragma: no cover
                        vector = None
            if vector is None:
                rng = np.random.default_rng(abs(hash((modality, self.output_dim))) % (2**32))
                vector = rng.normal(size=self.output_dim)
            anchors[modality] = self._resize(_normalise(vector), self.output_dim)
        return anchors

    def _initialise_linear_head(self, modality: str, input_dim: int) -> np.ndarray:
        seed = abs(hash((modality, input_dim, self.output_dim))) % (2**32)
        rng = np.random.default_rng(seed)
        weights = rng.normal(scale=1.0 / max(1, input_dim), size=(input_dim, self.output_dim))
        return weights.astype(float)

    def _initialise_audio_head(self, input_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        seed = (input_dim * 7919 + self.output_dim * 1543) % (2**32)
        rng = np.random.default_rng(seed)
        weights = rng.normal(scale=1.0 / max(1, input_dim), size=(input_dim, self.output_dim))
        bias = rng.normal(scale=0.05, size=(self.output_dim,))
        return weights.astype(float), bias.astype(float)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _resize(self, vector: np.ndarray, target: int) -> np.ndarray:
        if vector.size == target:
            return vector.astype(float, copy=False)
        if vector.size > target:
            return vector[:target].astype(float, copy=False)
        padded = np.zeros(target, dtype=float)
        padded[: vector.size] = vector.astype(float, copy=False)
        return padded

    @staticmethod
    def _canonical_modality(modality: Optional[str]) -> str:
        if not modality:
            return "generic"
        modality = modality.lower()
        aliases = {
            "visual": "vision",
            "image": "vision",
            "language": "text",
        }
        return aliases.get(modality, modality)

    @staticmethod
    def _extract_similarity(metadata: Dict[str, Any]) -> float:
        value = metadata.get("similarity") or metadata.get("confidence") or 0.0
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            try:
                value = max(float(v) for v in value)
            except (TypeError, ValueError):
                value = 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


__all__ = ["CrossModalTransformer"]
