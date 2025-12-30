"""Prototype differentiable neural computer (DNC) module."""

from __future__ import annotations

import numpy as np


class DifferentiableNeuralComputer:
    """Very small differentiable memory module.

    This is *not* a full implementation of the DNC paper but provides a
    minimal interface for experimentation and extension.  The memory is
    represented as a ``memory_size`` x ``word_size`` matrix and simple dot
    product similarity is used for addressing.
    """

    def __init__(self, memory_size: int, word_size: int) -> None:
        self.memory = np.zeros((memory_size, word_size), dtype=float)

    def read(self, key: np.ndarray) -> np.ndarray:
        """Return the memory row most similar to ``key``."""
        weights = self.memory @ key
        idx = int(np.argmax(weights))
        return self.memory[idx]

    def write(self, key: np.ndarray, value: np.ndarray) -> None:
        """Write ``value`` to the location most similar to ``key``."""
        weights = self.memory @ key
        idx = int(np.argmax(weights))
        self.memory[idx] = value

    def reset(self) -> None:
        """Clear the memory matrix."""
        self.memory.fill(0)
