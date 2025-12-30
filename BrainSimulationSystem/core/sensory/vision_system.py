from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .utils import resample_1d


class VisionSystem:
    """Simplified retina → (LGN-like) thalamic vector encoder.

    The output is a 1D vector intended to be injected into a cortical column's
    thalamic nucleus via ``column.process_sensory_input``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.thalamic_size = int(cfg.get("thalamic_size", 200))
        self.edge_gain = float(cfg.get("edge_gain", 1.0))
        self.intensity_gain = float(cfg.get("intensity_gain", 0.5))
        self.downsample_shape: Tuple[int, int] = tuple(cfg.get("downsample_shape", (16, 16)))  # type: ignore[assignment]

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3:
            # RGB/RGBA -> luminance
            channels = image.shape[-1]
            if channels >= 3:
                rgb = image[..., :3]
                weights = np.array([0.2989, 0.5870, 0.1140], dtype=float)
                gray = np.tensordot(rgb, weights, axes=([-1], [0]))
            else:
                gray = image[..., 0]
        else:
            gray = image.reshape(-1)
        gray = np.asarray(gray, dtype=float)
        if gray.size == 0:
            return np.zeros((0, 0), dtype=float)
        if gray.ndim != 2:
            side = int(round(np.sqrt(gray.size)))
            if side <= 0:
                return np.zeros((0, 0), dtype=float)
            gray = gray[: side * side].reshape(side, side)
        return gray

    @staticmethod
    def _normalize_01(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr.astype(float, copy=False)
        arr = arr.astype(float, copy=False)
        max_val = float(np.max(arr))
        if max_val > 1.5:  # likely 0..255
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)
        return arr

    @staticmethod
    def _resize_bilinear(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        image = np.asarray(image, dtype=float)
        if image.size == 0 or target_h <= 0 or target_w <= 0:
            return np.zeros((max(0, target_h), max(0, target_w)), dtype=float)
        h, w = image.shape
        if h == target_h and w == target_w:
            return image
        y_old = np.linspace(0.0, 1.0, num=h, endpoint=True)
        x_old = np.linspace(0.0, 1.0, num=w, endpoint=True)
        y_new = np.linspace(0.0, 1.0, num=target_h, endpoint=True)
        x_new = np.linspace(0.0, 1.0, num=target_w, endpoint=True)

        # Interpolate along x for each original row.
        tmp = np.zeros((h, target_w), dtype=float)
        for yi in range(h):
            tmp[yi] = np.interp(x_new, x_old, image[yi])
        # Interpolate along y for each new column.
        out = np.zeros((target_h, target_w), dtype=float)
        for xi in range(target_w):
            out[:, xi] = np.interp(y_new, y_old, tmp[:, xi])
        return out

    @staticmethod
    def _edge_map(gray: np.ndarray) -> np.ndarray:
        gray = np.asarray(gray, dtype=float)
        if gray.size == 0:
            return gray
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        if gray.shape[1] >= 3:
            gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        if gray.shape[0] >= 3:
            gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        edges = np.sqrt(gx * gx + gy * gy)
        return edges

    def encode(self, image: Any) -> np.ndarray:
        arr = np.asarray(image)
        gray = self._normalize_01(self._to_grayscale(arr))

        target_h, target_w = self.downsample_shape
        resized = self._resize_bilinear(gray, int(target_h), int(target_w))
        edges = self._edge_map(resized)

        # Combine intensity + edges into a single thalamic drive vector.
        flat = (self.intensity_gain * resized.reshape(-1) + self.edge_gain * edges.reshape(-1))
        vec = resample_1d(flat, self.thalamic_size)
        return np.clip(vec, 0.0, 1.0)

