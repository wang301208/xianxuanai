"""Basal ganglia inspired action selection utilities.

The implementation is intentionally lightweight: it maps action values to a
winner-take-all style decision using a softmax competition whose temperature is
modulated by dopamine-like signals (higher dopamine -> more exploitation).

This module is designed to be used by the existing ``DecisionProcess`` as an
optional decision backend without pulling in the heavier large-scale BG models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class BGSelection:
    action: Any
    confidence: float
    probabilities: Dict[str, float]
    metadata: Dict[str, Any]


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _extract_dopamine(context: Optional[Dict[str, Any]]) -> float:
    if not isinstance(context, dict):
        return 1.0

    for key in ("dopamine", "dopamine_level"):
        if key in context:
            return _safe_float(context.get(key), 1.0)

    neuromod = context.get("neuromodulation")
    if isinstance(neuromod, dict) and "dopamine" in neuromod:
        return _safe_float(neuromod.get("dopamine"), 1.0)

    emotion_state = context.get("emotion_state")
    if isinstance(emotion_state, dict):
        neuros = emotion_state.get("neuromodulators")
        if isinstance(neuros, dict) and "dopamine" in neuros:
            return _safe_float(neuros.get("dopamine"), 1.0)

    return 1.0


class ActionSelectionBG:
    """Winner-take-all action selection with dopamine-modulated competition."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = params or {}
        self.base_temperature = _safe_float(self.params.get("temperature", 1.0), 1.0)
        self.min_temperature = _safe_float(self.params.get("min_temperature", 0.05), 0.05)
        self.max_temperature = _safe_float(self.params.get("max_temperature", 5.0), 5.0)
        self.dopamine_gain = _safe_float(self.params.get("dopamine_gain", 0.7), 0.7)
        self.noise_std = _safe_float(self.params.get("noise_std", 0.0), 0.0)
        self.deterministic = bool(self.params.get("deterministic", True))
        seed = self.params.get("seed")
        self._rng = np.random.default_rng(int(seed)) if seed is not None else np.random.default_rng()

    def _effective_temperature(self, dopamine: float) -> float:
        # dopamine in ~[0, 2] -> shrink temperature as dopamine rises
        dopamine_value = float(np.clip(_safe_float(dopamine, 1.0), 0.0, 2.0))
        shrink = 1.0 + self.dopamine_gain * dopamine_value
        temp = self.base_temperature / max(shrink, 1e-6)
        return float(np.clip(temp, self.min_temperature, self.max_temperature))

    def select(
        self,
        options: Sequence[Any],
        values: Sequence[float],
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> BGSelection:
        if not options:
            return BGSelection(action=None, confidence=0.0, probabilities={}, metadata={"reason": "no_options"})

        values_arr = np.asarray(list(values), dtype=float)
        if values_arr.size != len(options):
            values_arr = np.resize(values_arr, len(options))

        dopamine = _extract_dopamine(context)
        temperature = self._effective_temperature(dopamine)

        logits = values_arr / max(temperature, 1e-6)
        logits = logits - float(np.max(logits))
        if self.noise_std > 0.0:
            logits = logits + self._rng.normal(0.0, float(self.noise_std), size=logits.shape)

        exp_logits = np.exp(logits)
        denom = float(np.sum(exp_logits))
        if not np.isfinite(denom) or denom <= 0.0:
            probs = np.full(len(options), 1.0 / len(options), dtype=float)
        else:
            probs = exp_logits / denom

        if self.deterministic:
            idx = int(np.argmax(probs))
        else:
            idx = int(self._rng.choice(len(options), p=probs))

        selected = options[idx]
        confidence = float(probs[idx])

        probabilities = {str(opt): float(probs[i]) for i, opt in enumerate(options)}
        metadata = {"dopamine": float(dopamine), "temperature": float(temperature)}
        return BGSelection(action=selected, confidence=confidence, probabilities=probabilities, metadata=metadata)

