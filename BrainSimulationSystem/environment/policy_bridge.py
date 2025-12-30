"""Map cognitive-level decisions to low-level environment actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from .base import PerceptionPacket


@dataclass(frozen=True)
class HighLevelDecision:
    """Canonical representation of a cognitive decision."""

    intent: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


DecisionSource = Callable[[PerceptionPacket], Mapping[str, Any] | HighLevelDecision | str]
ActionMapper = Callable[[Any, HighLevelDecision], Any]


class HierarchicalPolicyBridge:
    """Bridge cognition outputs, feature encoding, and RL policy actions."""

    def __init__(
        self,
        *,
        decision_fn: DecisionSource,
        action_space: Sequence[Any],
        feature_dim: int,
        action_mapper: ActionMapper | None = None,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if not action_space:
            raise ValueError("action_space must contain at least one entry")

        self._decision_fn = decision_fn
        self._feature_dim = int(feature_dim)
        self._action_space = list(action_space)
        self._action_mapper = action_mapper or (lambda action, _decision: action)
        self._intent_index: Dict[str, int] = {}

    # ------------------------------------------------------------------ #
    def select_action(
        self,
        packet: PerceptionPacket,
        rl_agent: Any,
    ) -> Tuple[Any, int, np.ndarray, HighLevelDecision, Dict[str, float]]:
        """Encode state, query the RL policy, and map to environment action."""

        decision = self.decision_from_packet(packet)
        obs_vector = self.encode(packet, decision)
        action_idx, metadata = rl_agent.select_action(obs_vector)
        action_idx = int(max(0, min(len(self._action_space) - 1, action_idx)))
        action_token = self._action_space[action_idx]
        low_level_action = self._action_mapper(action_token, decision)
        return low_level_action, action_idx, obs_vector, decision, metadata

    @property
    def action_space(self) -> Tuple[Any, ...]:
        """Return the underlying discrete action space."""

        return tuple(self._action_space)

    def map_action(self, action_token: Any, decision: HighLevelDecision) -> Any:
        """Map an action token from :attr:`action_space` into a low-level environment action."""

        return self._action_mapper(action_token, decision)

    def decision_from_packet(self, packet: PerceptionPacket) -> HighLevelDecision:
        """Return the canonical decision extracted from ``packet``."""

        return self._coerce_decision(self._decision_fn(packet))

    # ------------------------------------------------------------------ #
    def encode(self, packet: PerceptionPacket, decision: HighLevelDecision) -> np.ndarray:
        """Convert the multimodal perception packet into a fixed-size vector."""

        features: list[float] = []
        if packet.state_vector:
            features.extend(float(x) for x in packet.state_vector)
        if packet.proprioception:
            features.extend(float(v) for v in packet.proprioception.values())
        if packet.rewards:
            features.append(float(next(iter(packet.rewards.values()))))
        features.append(float(decision.confidence))
        features.append(self._intent_embedding(decision.intent))
        features.extend(self._metadata_projection(decision.metadata))

        vector = np.zeros(self._feature_dim, dtype=np.float32)
        length = min(len(features), self._feature_dim)
        vector[:length] = features[:length]
        return vector

    # ------------------------------------------------------------------ #
    def _metadata_projection(self, metadata: Mapping[str, Any]) -> Sequence[float]:
        projected: list[float] = []
        for key, value in sorted(metadata.items()):
            try:
                projected.append(float(value))
            except (TypeError, ValueError):
                projected.append(float(hash((key, str(value))) % 1000) / 1000.0)
        return projected

    def _intent_embedding(self, intent: str) -> float:
        if intent not in self._intent_index:
            self._intent_index[intent] = len(self._intent_index) + 1
        idx = self._intent_index[intent]
        return float(idx) / float(max(1, len(self._intent_index)))

    def _coerce_decision(
        self,
        payload: Mapping[str, Any] | HighLevelDecision | str,
    ) -> HighLevelDecision:
        if isinstance(payload, HighLevelDecision):
            return payload
        if isinstance(payload, str):
            return HighLevelDecision(intent=payload)
        if isinstance(payload, Mapping):
            data = dict(payload)
            intent = str(data.pop("intent", data.pop("intention", "observe")))
            confidence = float(data.pop("confidence", 1.0))
            return HighLevelDecision(intent=intent, confidence=confidence, metadata=dict(data))
        raise TypeError(f"Unsupported decision payload: {type(payload)!r}")


__all__ = ["HierarchicalPolicyBridge", "HighLevelDecision"]
