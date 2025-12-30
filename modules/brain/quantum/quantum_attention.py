"""Quantum attention mechanisms for amplitude re-weighting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .quantum_cognition import SuperpositionState


@dataclass
class QuantumAttention:
    """Applies focus weights to a superposition state."""

    def apply(self, state: SuperpositionState, focus: Mapping[str, float]) -> SuperpositionState:
        amps = {basis: amp * focus.get(basis, 1.0) for basis, amp in state.amplitudes.items()}
        return SuperpositionState(amps)


__all__ = ["QuantumAttention"]
