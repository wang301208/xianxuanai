"""Quantum reasoning via amplitude interference."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Mapping

import numpy as np


@dataclass
class QuantumReasoning:
    """Combine amplitudes to produce interference-based probabilities."""

    def interference(self, options: Mapping[str, Iterable[complex]]) -> Dict[str, float]:
        amps = {label: sum(values) for label, values in options.items()}
        norm = math.sqrt(sum(abs(a) ** 2 for a in amps.values()))
        if not np.isclose(norm, 1):
            amps = {k: v / norm for k, v in amps.items()}
        return {label: float(abs(a) ** 2) for label, a in amps.items()}


__all__ = ["QuantumReasoning"]
