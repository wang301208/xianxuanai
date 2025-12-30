from __future__ import annotations

from typing import Any

import numpy as np


def to_1d_float_array(value: Any) -> np.ndarray:
    """Convert the input into a 1D float array (best-effort)."""
    if value is None:
        return np.zeros(0, dtype=float)
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return np.zeros(0, dtype=float)
    if arr.size == 0:
        return np.zeros(0, dtype=float)
    return arr.reshape(-1).astype(float, copy=False)


def resample_1d(vector: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a 1D vector to `target_len` using linear interpolation."""
    target_len = int(target_len)
    if target_len <= 0:
        return np.zeros(0, dtype=float)
    vec = np.asarray(vector, dtype=float).reshape(-1)
    if vec.size == 0:
        return np.zeros(target_len, dtype=float)
    if vec.size == target_len:
        return vec.astype(float, copy=False)
    if vec.size == 1:
        return np.full(target_len, float(vec[0]), dtype=float)

    x_old = np.linspace(0.0, 1.0, num=int(vec.size), endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=True)
    return np.interp(x_new, x_old, vec).astype(float, copy=False)

