from __future__ import annotations

"""Lightweight neural architecture search helpers."""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional


@dataclass(frozen=True)
class NASParameter:
    """Search space description for a single architectural parameter."""

    name: str
    min_value: float
    max_value: float
    step: float = 1.0
    dtype: str = "int"  # "int" or "float"
    default: Optional[float] = None

    def clamp(self, value: float) -> float:
        lo = min(self.min_value, self.max_value)
        hi = max(self.min_value, self.max_value)
        return max(lo, min(hi, value))

    def quantise(self, value: float) -> float:
        step = max(abs(self.step), 1e-6)
        snapped = round(value / step) * step
        if self.dtype == "int":
            return float(int(snapped))
        return snapped

    def apply(self, architecture: Dict[str, float]) -> None:
        current = architecture.get(self.name)
        if current is None:
            if self.default is not None:
                architecture[self.name] = self.quantise(self.clamp(self.default))
            return
        architecture[self.name] = self.quantise(self.clamp(float(current)))


class NASMutationSpace:
    """Project GA mutations back into a valid neural architecture space."""

    def __init__(self, parameters: Mapping[str, NASParameter]) -> None:
        self.parameters: Dict[str, NASParameter] = dict(parameters)

    def postprocess(self, architecture: Dict[str, float]) -> None:
        """Clamp and quantise architecture parameters in-place."""

        for param in self.parameters.values():
            param.apply(architecture)

    def defaults(self) -> Dict[str, float]:
        """Return the default architecture values defined by the search space."""

        defaults: Dict[str, float] = {}
        for param in self.parameters.values():
            value = param.default
            if value is None:
                value = param.quantise(param.clamp(param.min_value))
            defaults[param.name] = value
        return defaults

