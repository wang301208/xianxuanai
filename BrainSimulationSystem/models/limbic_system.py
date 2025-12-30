"""Limbic circuit simulation helpers.

This module provides lightweight, architecture-compatible models for two key
limbic subsystems:

- Amygdala-like threat appraisal that produces a fast arousal/threat drive.
- Ventral striatum (NAc/VTA-like) reward prediction error dynamics that gate
  dopamine-like neuromodulatory signals.

The intent is not to be a biophysically detailed model, but to offer a clean
interface that can plug into ``BrainSimulation.step`` and provide structured
signals (threat, RPE, neuromodulators, recommended biases) without external
dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clip(value: float, lo: float, hi: float) -> float:
    return float(np.clip(float(value), float(lo), float(hi)))


@dataclass
class AmygdalaState:
    threat_level: float = 0.0
    mode: str = "baseline"


class AmygdalaCircuit:
    """Simple threat appraisal with a fast response mode."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.decay = _clip(_safe_float(cfg.get("decay", 0.9), 0.9), 0.0, 0.9999)
        self.gain = _clip(_safe_float(cfg.get("gain", 1.0), 1.0), 0.0, 10.0)
        self.freeze_threshold = _clip(_safe_float(cfg.get("freeze_threshold", 0.8), 0.8), 0.0, 1.0)
        self.flee_threshold = _clip(_safe_float(cfg.get("flee_threshold", 0.55), 0.55), 0.0, 1.0)
        self.caution_threshold = _clip(_safe_float(cfg.get("caution_threshold", 0.25), 0.25), 0.0, 1.0)
        self.norepinephrine_gain = _clip(_safe_float(cfg.get("norepinephrine_gain", 0.8), 0.8), 0.0, 5.0)
        self.state = AmygdalaState()

    def update(self, threat_signal: Optional[float], *, dt: float) -> Dict[str, Any]:
        raw = _clip(_safe_float(threat_signal, 0.0) * self.gain, 0.0, 1.0)
        # Discrete leaky integration. ``dt`` is in milliseconds in this codebase,
        # so we keep the update intentionally simple and stable.
        prior = float(self.state.threat_level)
        threat = prior * float(self.decay) + raw * (1.0 - float(self.decay))
        threat = _clip(threat, 0.0, 1.0)
        self.state.threat_level = threat

        if threat >= self.freeze_threshold:
            mode = "freeze"
        elif threat >= self.flee_threshold:
            mode = "flee"
        elif threat >= self.caution_threshold:
            mode = "caution"
        else:
            mode = "baseline"
        self.state.mode = mode

        sympathetic_arousal = _clip(0.2 + 0.8 * threat, 0.0, 1.0)
        norepinephrine = _clip(1.0 + self.norepinephrine_gain * threat, 0.0, 2.0)

        return {
            "threat_level": threat,
            "mode": mode,
            "sympathetic_arousal": sympathetic_arousal,
            "neuromodulators": {
                "norepinephrine": norepinephrine,
            },
            "action_bias": {
                "avoid": _clip(0.2 + 0.8 * threat, 0.0, 1.0),
                "explore": _clip(1.0 - threat, 0.0, 1.0),
            },
            "dt": float(dt),
            "source": "amygdala",
        }


@dataclass
class VentralStriatumState:
    expected_reward: float = 0.0
    dopamine_level: float = 1.0
    last_rpe: float = 0.0


class VentralStriatumCircuit:
    """Reward prediction error and dopamine-like modulation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.expected_reward_decay = _clip(_safe_float(cfg.get("expected_reward_decay", 0.9), 0.9), 0.0, 0.9999)
        self.rpe_clip = _clip(_safe_float(cfg.get("rpe_clip", 1.0), 1.0), 0.0, 10.0)
        self.dopamine_baseline = _clip(_safe_float(cfg.get("dopamine_baseline", 1.0), 1.0), 0.0, 2.0)
        self.dopamine_gain = _clip(_safe_float(cfg.get("dopamine_gain", 0.6), 0.6), 0.0, 5.0)
        self.novelty_gain = _clip(_safe_float(cfg.get("novelty_gain", 0.15), 0.15), 0.0, 5.0)
        self.state = VentralStriatumState(expected_reward=0.0, dopamine_level=self.dopamine_baseline, last_rpe=0.0)

    def update(
        self,
        *,
        reward: Optional[float],
        predicted_reward: Optional[float],
        novelty: Optional[float] = None,
    ) -> Dict[str, Any]:
        pred = None if predicted_reward is None else _safe_float(predicted_reward, 0.0)
        if pred is not None:
            self.state.expected_reward = float(self.state.expected_reward) * float(self.expected_reward_decay) + pred * (
                1.0 - float(self.expected_reward_decay)
            )

        reward_value = None if reward is None else _safe_float(reward, 0.0)
        rpe = 0.0
        if reward_value is not None:
            rpe = reward_value - float(self.state.expected_reward)
        rpe = _clip(rpe, -self.rpe_clip, self.rpe_clip)
        self.state.last_rpe = float(rpe)

        novelty_value = _clip(_safe_float(novelty, 0.0), 0.0, 1.0) if novelty is not None else 0.0
        dopamine = self.dopamine_baseline + self.dopamine_gain * rpe + self.novelty_gain * novelty_value
        dopamine = _clip(dopamine, 0.0, 2.0)
        self.state.dopamine_level = dopamine

        return {
            "reward": reward_value,
            "predicted_reward": pred,
            "expected_reward": float(self.state.expected_reward),
            "rpe": float(rpe),
            "novelty": float(novelty_value),
            "neuromodulators": {
                "dopamine": float(dopamine),
            },
            "source": "ventral_striatum",
        }


class LimbicSystem:
    """Wrapper combining amygdala and ventral striatum circuits."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.apply_to_emotion_system = bool(cfg.get("apply_to_emotion_system", True))
        self.amygdala = AmygdalaCircuit(cfg.get("amygdala", {}))
        self.ventral_striatum = VentralStriatumCircuit(
            cfg.get("ventral_striatum", cfg.get("nucleus_accumbens", {}))
        )

    def update(
        self,
        *,
        reward: Optional[float],
        predicted_reward: Optional[float],
        threat: Optional[float],
        novelty: Optional[float],
        dt: float,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}

        amygdala_state = self.amygdala.update(threat, dt=float(dt))
        vs_state = self.ventral_striatum.update(
            reward=reward,
            predicted_reward=predicted_reward,
            novelty=novelty,
        )

        neuromodulators: Dict[str, float] = {}
        for src in (amygdala_state, vs_state):
            mods = src.get("neuromodulators")
            if isinstance(mods, dict):
                for key, value in mods.items():
                    neuromodulators[str(key)] = _safe_float(value, neuromodulators.get(str(key), 0.0))

        # Provide a generic decision modulation hint; consumers may ignore.
        threat_level = _safe_float(amygdala_state.get("threat_level"), 0.0)
        dopamine_level = _safe_float(neuromodulators.get("dopamine", 1.0), 1.0)
        temperature_scale = _clip(1.0 + 0.4 * max(0.0, dopamine_level - 1.0) - 0.8 * threat_level, 0.2, 2.5)

        return {
            "enabled": True,
            "amygdala": amygdala_state,
            "ventral_striatum": vs_state,
            "neuromodulators": neuromodulators,
            "decision_bias": {
                "temperature_scale": float(temperature_scale),
                "threat_level": float(threat_level),
            },
        }


__all__ = ["AmygdalaCircuit", "VentralStriatumCircuit", "LimbicSystem"]

