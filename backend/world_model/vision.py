"""Utilities for storing visual observations in the world model.

This module provides a small in-memory store that keeps track of visual
observations associated with agents. Observations can be raw image "tensors"
(``torch.Tensor`` or ``numpy.ndarray``) or any pre-computed feature vectors
such as those produced by CLIP. The store keeps the provided objects so that
callers can retrieve them later. If PyTorch is unavailable at runtime the
module falls back to lightweight NumPy based fusion so the API remains usable
in minimal environments.
"""

from typing import Any, Dict, Optional, Sequence, TYPE_CHECKING

import numpy as np

try:  # pragma: no cover - exercised indirectly when torch is installed
    import torch
    from torch import nn
except Exception:  # pragma: no cover - runtime environments without torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from torch import Tensor
else:  # Fallback type alias when torch is not available
    Tensor = Any


def _to_numpy(value: Any) -> np.ndarray:
    """Convert ``value`` into a NumPy array without importing torch eagerly."""

    if torch is not None and isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _ensure_2d(array: np.ndarray) -> np.ndarray:
    """Ensure ``array`` has shape ``(batch, features)`` for aggregation."""

    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim == 0:
        return array.reshape(1, 1)
    return array.reshape(array.shape[0], -1)


def _vector_length(value: Any) -> int:
    """Best-effort inference of the feature dimension for ``value``."""

    if torch is not None and isinstance(value, torch.Tensor):
        return int(value.shape[-1]) if value.dim() > 0 else 1
    shape = getattr(value, "shape", None)
    if isinstance(shape, Sequence) and shape:
        return int(shape[-1])
    array = _to_numpy(value)
    return int(array.shape[-1]) if array.ndim > 0 else int(array.size or 1)


_BaseModule = nn.Module if nn is not None else object


class CrossModalAttention(_BaseModule):
    """Cross-modal attention with graceful fallback when torch is unavailable."""

    def __init__(self, embed_dim: int, num_heads: int = 8) -> None:
        if embed_dim <= 0:
            embed_dim = 1
        self.embed_dim = int(embed_dim)
        if nn is not None:
            actual_heads = num_heads if embed_dim % num_heads == 0 else 1
        else:
            actual_heads = 1
        super().__init__()
        self._torch_module: Optional[nn.MultiheadAttention]
        if nn is not None:
            self._torch_module = nn.MultiheadAttention(
                self.embed_dim, actual_heads, batch_first=True
            )
        else:
            self._torch_module = None

    def forward(self, vision_feat: Any, text_feat: Any) -> Any:
        if self._torch_module is not None and torch is not None:
            vision_tensor = torch.as_tensor(vision_feat)
            text_tensor = torch.as_tensor(text_feat)
            if vision_tensor.dim() == 1:
                vision_tensor = vision_tensor.unsqueeze(0).unsqueeze(0)
            elif vision_tensor.dim() == 2:
                vision_tensor = vision_tensor.unsqueeze(0)
            if text_tensor.dim() == 1:
                text_tensor = text_tensor.unsqueeze(0).unsqueeze(0)
            elif text_tensor.dim() == 2:
                text_tensor = text_tensor.unsqueeze(0)
            text_attended, _ = self._torch_module(text_tensor, vision_tensor, vision_tensor)
            vision_attended, _ = self._torch_module(vision_tensor, text_tensor, text_tensor)
            unified = (text_attended.mean(dim=1) + vision_attended.mean(dim=1)) / 2
            return unified.squeeze(0)
        return self._fallback_forward(vision_feat, text_feat)

    __call__ = forward

    def _fallback_forward(self, vision_feat: Any, text_feat: Any) -> np.ndarray:
        vision_arr = _ensure_2d(_to_numpy(vision_feat))
        text_arr = _ensure_2d(_to_numpy(text_feat))
        if vision_arr.size == 0 or text_arr.size == 0:
            return np.zeros(0, dtype=float)
        embed_dim = min(vision_arr.shape[-1], text_arr.shape[-1])
        if embed_dim <= 0:
            return np.zeros(0, dtype=float)
        vision_mean = vision_arr[:, :embed_dim].mean(axis=0)
        text_mean = text_arr[:, :embed_dim].mean(axis=0)
        return (vision_mean + text_mean) / 2.0


class VisionStore:
    """In-memory storage for visual observations."""

    def __init__(self) -> None:
        self._images: Dict[str, Any] = {}
        self._features: Dict[str, Any] = {}
        self._vit_features: Dict[str, Any] = {}
        self._text: Dict[str, Any] = {}
        self._unified: Dict[str, Any] = {}
        self._attn: Optional[CrossModalAttention] = None

    def ingest(
        self,
        agent_id: str,
        image: Optional[Any] = None,
        features: Optional[Any] = None,
        vit_features: Optional[Any] = None,
        text: Optional[Any] = None,
    ) -> Optional[Any]:
        """Store an observation for ``agent_id`` and optionally compute fusion."""

        if image is not None:
            self._images[agent_id] = image
        if features is not None:
            self._features[agent_id] = features
        if vit_features is not None:
            self._vit_features[agent_id] = vit_features
        if text is not None:
            self._text[agent_id] = text

        unified: Optional[Any] = None
        vision_feat = features if features is not None else vit_features
        if text is not None and vision_feat is not None:
            embed_dim = _vector_length(vision_feat)
            if self._attn is None or getattr(self._attn, "embed_dim", None) != embed_dim:
                self._attn = CrossModalAttention(embed_dim)
            unified = self._attn(vision_feat, text)
            self._unified[agent_id] = unified

        return unified

    def get(self, agent_id: str) -> Dict[str, Any]:
        """Return the latest observation for ``agent_id``."""

        return {
            "image": self._images.get(agent_id),
            "features": self._features.get(agent_id),
            "vit_features": self._vit_features.get(agent_id),
            "text": self._text.get(agent_id),
            "unified": self._unified.get(agent_id),
        }

    def all(self) -> Dict[str, Dict[str, Any]]:
        """Return a snapshot of all stored observations."""

        keys = (
            set(self._images)
            | set(self._features)
            | set(self._vit_features)
            | set(self._text)
            | set(self._unified)
        )
        return {
            agent: {
                "image": self._images.get(agent),
                "features": self._features.get(agent),
                "vit_features": self._vit_features.get(agent),
                "text": self._text.get(agent),
                "unified": self._unified.get(agent),
            }
            for agent in keys
        }


__all__ = ["VisionStore", "CrossModalAttention"]
