from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


class AdaptiveGainAttentionModule:
    """Simplified LC (norepinephrine) attention modulation model.

    Produces:
    - ``neuromodulators``: a dict suitable for ``SynapseManager.update_all_synapses``.
    - ``attention_gain``: a scalar gain that can be applied to sensory pathways.

    The implementation follows the spirit of Adaptive Gain Theory: tonic NE
    tracks uncertainty/arousal, while a phasic component responds to novelty
    and conflict/error signals.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(config or {})

        self.enabled = bool(cfg.get("enabled", False))

        self.tonic_level = float(cfg.get("tonic_level", 0.4))
        self.tonic_baseline = float(cfg.get("tonic_baseline", 0.4))
        self.tonic_tau_ms = float(cfg.get("tonic_tau_ms", 250.0))

        self.phasic_gain = float(cfg.get("phasic_gain", 0.35))
        self.min_release = float(cfg.get("min_release", 0.0))
        self.max_release = float(cfg.get("max_release", 1.5))

        self.optimal_release = float(cfg.get("optimal_release", 1.0))
        self.inverted_u_slope = float(cfg.get("inverted_u_slope", 0.6))
        self.min_attention_gain = float(cfg.get("min_attention_gain", 0.5))
        self.max_attention_gain = float(cfg.get("max_attention_gain", 2.0))

        weights = cfg.get("weights", {})
        if not isinstance(weights, dict):
            weights = {}
        self.novelty_weight = float(weights.get("novelty", 0.6))
        self.error_weight = float(weights.get("error", 0.5))
        self.conflict_weight = float(weights.get("conflict", 0.3))
        self.uncertainty_weight = float(weights.get("uncertainty", 0.35))
        self.arousal_weight = float(weights.get("arousal", 0.25))
        self.stress_weight = float(weights.get("stress", 0.25))

        self._last_sensory_energy = 0.0
        self._last_release = float(self.tonic_level)
        self._last_attention_gain = 1.0

    def update(
        self,
        *,
        dt: float,
        spike_count: int,
        network_synchrony: float,
        sensory: Optional[Dict[str, Any]] = None,
        external_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {}

        dt_ms = float(dt)
        if not np.isfinite(dt_ms) or dt_ms <= 0.0:
            dt_ms = 1.0

        synchrony = float(network_synchrony)
        if not np.isfinite(synchrony):
            synchrony = 0.0
        synchrony = _clamp(synchrony, 0.0, 1.0)

        uncertainty = 1.0 - synchrony

        sensory_energy = 0.0
        if isinstance(sensory, dict):
            for payload in sensory.values():
                if not isinstance(payload, dict):
                    continue
                try:
                    sensory_energy += float(payload.get("energy", 0.0))
                except (TypeError, ValueError):
                    continue

        novelty = 0.0
        if sensory_energy > 0.0:
            delta = max(0.0, sensory_energy - self._last_sensory_energy)
            novelty = delta / max(1e-6, abs(self._last_sensory_energy))
        novelty = _clamp(novelty, 0.0, 1.0)
        self._last_sensory_energy = float(sensory_energy)

        inputs = external_inputs if isinstance(external_inputs, dict) else {}
        arousal = inputs.get("arousal")
        stress = inputs.get("stress")
        error_signal = inputs.get("prediction_error", inputs.get("error_signal", 0.0))
        conflict = inputs.get("conflict", inputs.get("decision_conflict", 0.0))

        def _safe_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        arousal_value = _clamp(_safe_float(arousal, 0.5), 0.0, 1.0)
        stress_value = _clamp(_safe_float(stress, 0.0), 0.0, 1.0)
        error_value = _clamp(abs(_safe_float(error_signal, 0.0)), 0.0, 1.0)
        conflict_value = _clamp(_safe_float(conflict, 0.0), 0.0, 1.0)

        tonic_target = (
            self.tonic_baseline
            + self.uncertainty_weight * uncertainty
            + self.arousal_weight * (arousal_value - 0.5)
            - self.stress_weight * stress_value
        )
        tonic_target = _clamp(tonic_target, 0.0, self.max_release)

        tau = max(1.0, float(self.tonic_tau_ms))
        alpha = _clamp(dt_ms / tau, 0.0, 1.0)
        self.tonic_level = float(self.tonic_level + alpha * (tonic_target - self.tonic_level))
        self.tonic_level = _clamp(self.tonic_level, 0.0, self.max_release)

        phasic = (
            self.novelty_weight * novelty
            + self.error_weight * error_value
            + self.conflict_weight * conflict_value
        )
        phasic = _clamp(phasic, 0.0, 1.0)

        release = self.tonic_level + self.phasic_gain * phasic
        release = _clamp(release, self.min_release, self.max_release)

        deviation = abs(release - self.optimal_release)
        efficiency = 1.0 - self.inverted_u_slope * deviation
        efficiency = _clamp(efficiency, 0.0, 1.0)

        attention_gain = self.min_attention_gain + (self.max_attention_gain - self.min_attention_gain) * efficiency
        attention_gain = _clamp(attention_gain, self.min_attention_gain, self.max_attention_gain)

        self._last_release = float(release)
        self._last_attention_gain = float(attention_gain)

        return {
            "neuromodulators": {"norepinephrine": float(release)},
            "norepinephrine": float(release),
            "attention_gain": float(attention_gain),
            "tonic_level": float(self.tonic_level),
            "phasic_component": float(phasic),
            "novelty": float(novelty),
            "uncertainty": float(uncertainty),
            "spike_count": int(spike_count),
            "network_synchrony": float(synchrony),
        }


__all__ = ["AdaptiveGainAttentionModule"]

