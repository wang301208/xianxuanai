"""Abstract interface for hardware backends."""
from __future__ import annotations

from abc import ABC, abstractmethod


class DeviceBackend(ABC):
    """Abstract backend for numeric array operations.

    Implementations provide memory transfer primitives and kernel launches
    for their respective devices.
    """

    name: str = "abstract"

    @abstractmethod
    def to_device(self, array):
        """Transfer ``array`` to device memory."""

    @abstractmethod
    def from_device(self, array):
        """Transfer ``array`` from device memory to host."""

    @abstractmethod
    def matmul(self, a, b):
        """Return the matrix multiplication of ``a`` and ``b`` on device."""
