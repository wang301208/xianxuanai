"""CPU backend using NumPy."""
from __future__ import annotations

import numpy as np

from .base import DeviceBackend


class CPUBackend(DeviceBackend):
    """Backend implementation using ``numpy`` on the host CPU."""

    name = "cpu"

    def to_device(self, array):
        return np.asarray(array)

    def from_device(self, array):
        return np.asarray(array)

    def matmul(self, a, b):
        return np.matmul(a, b)
