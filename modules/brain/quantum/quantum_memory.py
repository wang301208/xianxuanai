"""Quantum memory storage and superposition retrieval."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

from .quantum_cognition import SuperpositionState


@dataclass
class QuantumMemory:
    """Simple storage for :class:`SuperpositionState` objects."""

    storage: Dict[str, SuperpositionState]

    def __init__(self) -> None:
        self.storage = {}

    def store(self, key: str, state: SuperpositionState) -> None:
        self.storage[key] = state

    def retrieve(self, key: str) -> SuperpositionState:
        return self.storage[key]

    def superposition(self, weights: Mapping[str, complex]) -> SuperpositionState:
        combined: Dict[str, complex] = {}
        for key, amp in weights.items():
            state = self.storage[key]
            for basis, value in state.amplitudes.items():
                combined[basis] = combined.get(basis, 0) + amp * value
        return SuperpositionState(combined)


__all__ = ["QuantumMemory"]
