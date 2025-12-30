"""
Emotion and motivation system inspired by limbic reward mechanisms.

Simulates valence/arousal adjustments based on reward signals and external
stimuli, and exposes a motivation drive that can be used by higher-level
planning components.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import numpy as np


@dataclass
class EmotionMotivationConfig:
    """Configuration for the emotion and motivation system."""

    baseline_valence: float = 0.0
    baseline_arousal: float = 0.3
    learning_rate: float = 0.2
    decay: float = 0.95
    motivation_gain: float = 0.5
    empathy_weight: float = 0.2


@dataclass
class EmotionState:
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.5
    timestamp: float = field(default_factory=time.time)


class EmotionMotivationSystem:
    """Tracks emotional state and derives motivation drives."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = EmotionMotivationConfig(**(config or {}))
        self.state = EmotionState(
            valence=self.config.baseline_valence,
            arousal=self.config.baseline_arousal,
            dominance=0.5,
        )
        self.last_reward = 0.0
        self.motivation_vector: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def update(
        self,
        reward: Optional[float] = None,
        stimuli: Optional[Dict[str, Any]] = None,
        empathy_input: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        stimuli = stimuli or {}
        empathy_input = empathy_input or {}

        if reward is not None:
            delta = self.config.learning_rate * float(reward)
            self.state.valence = np.clip(self.state.valence * self.config.decay + delta, -1.0, 1.0)
            self.state.arousal = np.clip(self.state.arousal + abs(delta) * 0.5, 0.0, 1.0)
            self.last_reward = reward

        if "threat" in stimuli:
            threat = float(stimuli["threat"])
            self.state.arousal = np.clip(self.state.arousal + threat * 0.3, 0.0, 1.0)
            self.state.valence = np.clip(self.state.valence - threat * 0.2, -1.0, 1.0)

        if "pleasure" in stimuli:
            pleasure = float(stimuli["pleasure"])
            self.state.valence = np.clip(self.state.valence + pleasure * 0.2, -1.0, 1.0)

        if empathy_input:
            perceived_valence = empathy_input.get("valence", 0.0)
            self.state.valence = np.clip(
                self.state.valence + self.config.empathy_weight * perceived_valence,
                -1.0,
                1.0,
            )

        self.state.timestamp = time.time()

        motivation = self._compute_motivation(stimuli.get("goals", []))
        self.motivation_vector = motivation
        return {
            "valence": self.state.valence,
            "arousal": self.state.arousal,
            "dominance": self.state.dominance,
            "last_reward": self.last_reward,
            "motivation": motivation,
        }

    def _compute_motivation(self, goals: Iterable[Any]) -> Dict[str, float]:
        motivation: Dict[str, float] = {}
        if not goals:
            base_drive = max(0.1, self.state.arousal * 0.5 + (self.state.valence + 1) / 2)
            motivation["explore"] = float(np.clip(base_drive, 0.0, 1.0))
            return motivation

        for goal in goals:
            goal_name = str(goal)
            valence_factor = (self.state.valence + 1.0) / 2.0
            arousal_factor = max(self.state.arousal, 0.1)
            drive = np.clip(self.config.motivation_gain * valence_factor * arousal_factor, 0.0, 1.0)
            motivation[goal_name] = float(drive)
        return motivation

    def get_state(self) -> Dict[str, Any]:
        return {
            "valence": self.state.valence,
            "arousal": self.state.arousal,
            "dominance": self.state.dominance,
            "last_reward": self.last_reward,
            "motivation": self.motivation_vector,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": {
                "valence": self.state.valence,
                "arousal": self.state.arousal,
                "dominance": self.state.dominance,
                "timestamp": self.state.timestamp,
            },
            "last_reward": self.last_reward,
            "motivation": self.motivation_vector,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        state = data.get("state", {})
        self.state = EmotionState(
            valence=float(state.get("valence", self.config.baseline_valence)),
            arousal=float(state.get("arousal", self.config.baseline_arousal)),
            dominance=float(state.get("dominance", 0.5)),
            timestamp=float(state.get("timestamp", time.time())),
        )
        self.last_reward = float(data.get("last_reward", 0.0))
        self.motivation_vector = {
            str(k): float(v) for k, v in data.get("motivation", {}).items()
        }
