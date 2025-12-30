"""Utilities to convert environment perceptions into curiosity stimuli."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple

import numpy as np

from BrainSimulationSystem.environment.base import PerceptionPacket


def _hashable_vector(vector: Sequence[float]) -> Tuple[float, ...]:
    return tuple(round(float(x), 3) for x in vector)


@dataclass
class CuriosityStimulusEncoder:
    """Incrementally derive novelty/complexity metrics for curiosity engines."""

    novelty_floor: float = 0.05
    novelty_decay: float = 0.7
    complexity_temperature: float = 1.0
    social_keys: Tuple[str, ...] = ("detected_faces", "speech_energy", "social_signal")
    _state_counts: Dict[Hashable, int] = field(default_factory=lambda: defaultdict(int))
    _last_complexity: float = 0.0

    def build(
        self,
        packet: PerceptionPacket,
        *,
        info: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a stimulus dict consumed by :class:`SocialCuriosityEngine`."""

        metadata = metadata or {}
        info = info or {}

        state_key = self._state_signature(packet, info)
        self._state_counts[state_key] += 1
        visits = self._state_counts[state_key]
        novelty = 1.0 / np.sqrt(visits)
        novelty = float(max(self.novelty_floor, novelty))

        complexity = self._compute_complexity(packet)
        uncertainty = float(abs(complexity - self._last_complexity))
        self._last_complexity = (self.novelty_decay * self._last_complexity) + (
            (1.0 - self.novelty_decay) * complexity
        )

        social_cues = self._extract_social_cues(packet, info)
        stimulus = {
            "novelty": novelty,
            "complexity": complexity,
            "uncertainty": uncertainty,
            "social_cues": social_cues,
            "social_context": bool(social_cues),
        }
        if "surprise" in metadata:
            try:
                stimulus["surprise"] = float(metadata["surprise"])
            except (TypeError, ValueError):
                pass
        return stimulus

    # ------------------------------------------------------------------ #
    def _state_signature(self, packet: PerceptionPacket, info: Mapping[str, Any]) -> Hashable:
        if packet.state_vector:
            return _hashable_vector(packet.state_vector)
        if packet.proprioception:
            return tuple(sorted((k, round(float(v), 3)) for k, v in packet.proprioception.items()))
        if info.get("state_id"):
            return info["state_id"]
        if packet.metadata:
            return tuple(sorted(packet.metadata.items()))
        return hash(
            (
                str(packet.vision)[:32],
                str(packet.audio)[:32],
                str(packet.depth_map)[:32],
            )
        )

    def _compute_complexity(self, packet: PerceptionPacket) -> float:
        if packet.state_vector:
            vec = np.asarray(packet.state_vector, dtype=np.float32)
            std = float(np.std(vec))
            scaled = np.tanh(std / max(1e-6, self.complexity_temperature))
            return max(0.0, min(1.0, scaled))
        if isinstance(packet.vision, (list, tuple, np.ndarray)):
            arr = np.asarray(packet.vision)
            if arr.size:
                texture = float(np.std(arr) / (np.mean(np.abs(arr)) + 1e-6))
                return max(0.0, min(1.0, np.tanh(texture)))
        return 0.2

    def _extract_social_cues(
        self,
        packet: PerceptionPacket,
        info: Mapping[str, Any],
    ) -> Dict[str, float]:
        cues: Dict[str, float] = {}
        base = {}
        for source in (packet.metadata or {}, info):
            social_data = source.get("social_cues") or source.get("social") or {}
            if isinstance(social_data, Mapping):
                base.update(social_data)
            for key in self.social_keys:
                value = source.get(key)
                if isinstance(value, (int, float)):
                    base.setdefault(key, float(value))

        if not base:
            return cues

        cues["eye_contact"] = float(base.get("detected_faces", 0.0))
        cues["expression"] = float(base.get("expression_valence", 0.0))
        cues["intonation"] = float(base.get("speech_energy", 0.0))
        cues["speech_rate"] = float(base.get("speech_rate", 0.0))
        cues["hand_movement"] = float(base.get("gesture_intensity", 0.0))
        cues["body_posture"] = float(base.get("body_pose", 0.0))

        return {k: max(0.0, min(1.0, v)) for k, v in cues.items()}


__all__ = ["CuriosityStimulusEncoder"]
