"""Simple imitation learning stubs (behavior cloning style)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence


@dataclass
class ImitationStats:
    losses: List[float]
    samples: int
    action_distribution: Dict[str, float]


class ImitationLearner:
    """Train a policy from demonstration trajectories."""

    def __init__(self) -> None:
        self.trained_steps = 0
        self._action_counts: Dict[str, int] = {}
        self._total_actions = 0

    def train(self, demonstrations: Iterable[Dict[str, Any]]) -> ImitationStats:
        """Consume demonstration trajectories and return simple stats."""

        losses: List[float] = []
        count = 0
        for demo in demonstrations:
            actions = _extract_actions(demo)
            if not actions:
                continue
            loss = self._update_from_actions(actions)
            losses.append(loss)
            count += 1
            self.trained_steps += 1

        return ImitationStats(
            losses=losses,
            samples=count,
            action_distribution=self.action_distribution(),
        )

    def action_distribution(self) -> Dict[str, float]:
        if not self._action_counts:
            return {}
        total = sum(self._action_counts.values())
        if total <= 0:
            return {}
        return {action: count / total for action, count in sorted(self._action_counts.items())}

    def _update_from_actions(self, actions: Sequence[str]) -> float:
        """Update the learner with a sequence of demonstrated actions.

        Returns a simple negative log-likelihood loss under an empirical,
        add-one-smoothed categorical model.
        """

        for action in actions:
            normalized = str(action).strip()
            if not normalized:
                continue
            self._action_counts[normalized] = self._action_counts.get(normalized, 0) + 1
            self._total_actions += 1

        vocabulary = max(len(self._action_counts), 1)
        denom = self._total_actions + vocabulary
        loss = 0.0
        counted = 0
        for action in actions:
            normalized = str(action).strip()
            if not normalized:
                continue
            prob = (self._action_counts.get(normalized, 0) + 1) / denom
            loss += -math.log(prob)
            counted += 1
        return loss / max(counted, 1)


def _extract_actions(demo: Mapping[str, Any]) -> List[str]:
    """Extract a flat list of action strings from a demonstration structure."""

    raw = demo.get("actions")
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw if str(item).strip()]
    single = demo.get("action")
    if isinstance(single, str) and single.strip():
        return [single.strip()]
    steps = demo.get("steps")
    if isinstance(steps, list):
        extracted: List[str] = []
        for step in steps:
            if isinstance(step, Mapping):
                action = step.get("action")
                if isinstance(action, str) and action.strip():
                    extracted.append(action.strip())
        return extracted
    return []


__all__ = ["ImitationLearner", "ImitationStats"]
