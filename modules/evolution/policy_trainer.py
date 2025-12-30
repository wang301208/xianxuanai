"""Utilities for training discrete policies from experience datasets.

This module implements a small REINFORCE-style trainer that can consume
experience tuples from in-memory buffers or external datasets. Dataset files
may be provided in CSV, JSON, or JSONL formats. Each record is mapped to a
``(state, action, reward)`` tuple using a transformation hook which can be
customised to handle arbitrary or nested schemas.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union


# ``Experience`` tuples may optionally include an ``agent_id`` as the first
# element allowing the trainer to manage multiple policies while sharing a
# common experience buffer.
Experience = Union[Tuple[int, str, str, float], Tuple[str, str, float]]


@dataclass
class PolicyTrainer:
    """Simple REINFORCE-style trainer for discrete policies.

    Experiences are tuples of ``(state, action, reward)``. The trainer can read
    additional experiences from CSV, JSON, or JSONL datasets and update a policy
    mapping states to action-preference values. Datasets may contain nested
    experience objects; a custom ``experience_transform`` callable can be
    supplied to extract the tuple from arbitrary structures.
    """

    dataset_path: Path
    learning_rate: float = 0.1
    policy: Dict[str, Dict[str, float]] = field(default_factory=dict)
    num_agents: int = 1
    experience_transform: Callable[[Dict[str, Any]], Experience] | None = None

    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_path)
        self._buffer: List[Experience] = []
        if self.num_agents == 1:
            self._policies: Dict[int, Dict[str, Dict[str, float]]] = {0: self.policy}
        else:
            self._policies = {i: {} for i in range(self.num_agents)}
            # maintain attribute ``policy`` for backwards compatibility
            self.policy = self._policies[0]
        if self.experience_transform is None:
            self.experience_transform = self._default_transform

    # ------------------------------------------------------------------
    # Experience management
    # ------------------------------------------------------------------
    def push_experience(self, state: str, action: str, reward: float, agent_id: int = 0) -> None:
        """Add an experience tuple to the shared in-memory buffer."""

        self._buffer.append((agent_id, state, action, reward))

    # Dataset loading --------------------------------------------------
    def _default_transform(self, obj: Dict[str, Any]) -> Experience:
        """Extract ``(agent_id, state, action, reward)`` from a dataset object.

        The default implementation supports both flat and nested ``{"experience"``
        ``: {...}}`` structures and assumes an ``agent_id`` field (defaulting to
        ``0`` when absent).
        """

        exp = obj.get("experience", obj)
        agent = int(exp.get("agent_id", 0))
        return agent, exp["state"], exp["action"], float(exp["reward"])

    def _append_from_obj(self, obj: Dict[str, Any], experiences: List[Experience]) -> None:
        try:
            experiences.append(self.experience_transform(obj))
        except Exception:
            # Silently skip malformed rows
            pass

    def _load_dataset(self) -> List[Experience]:
        experiences: List[Experience] = []
        if not self.dataset_path.exists():
            return experiences

        try:
            ext = self.dataset_path.suffix.lower()
            if ext == ".csv":
                with self.dataset_path.open(newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self._append_from_obj(row, experiences)
            elif ext == ".json":
                with self.dataset_path.open() as f:
                    data = json.load(f)
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    items = data.get("experiences", [data])
                else:
                    items = []
                for obj in items:
                    if isinstance(obj, dict):
                        self._append_from_obj(obj, experiences)
            elif ext == ".jsonl":
                with self.dataset_path.open() as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(obj, dict):
                            self._append_from_obj(obj, experiences)
        except Exception:
            pass

        return experiences

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------
    def _softmax(self, values: List[float]) -> List[float]:
        if not values:
            return []
        max_v = max(values)
        exps = [math.exp(v - max_v) for v in values]
        total = sum(exps) or 1.0
        return [e / total for e in exps]

    def update_policy(self) -> Dict[str, Dict[str, float]] | Dict[int, Dict[str, Dict[str, float]]]:
        """Update the policy or policies using buffered and dataset experiences.

        When ``num_agents`` is ``1`` the return value matches previous versions
        of this class (a single policy mapping). For multi-agent settings a
        dictionary keyed by ``agent_id`` is returned.
        """

        experiences = self._load_dataset() + self._buffer
        for exp in experiences:
            if len(exp) == 4:
                agent, state, action, reward = exp  # type: ignore[misc]
            else:
                state, action, reward = exp  # type: ignore[misc]
                agent = 0
            prefs = self._policies.setdefault(agent, {}).setdefault(state, {})
            prefs.setdefault(action, 0.0)
            actions = list(prefs.keys())
            probs = self._softmax([prefs[a] for a in actions])
            for name, prob in zip(actions, probs):
                indicator = 1.0 if name == action else 0.0
                prefs[name] += self.learning_rate * reward * (indicator - prob)
        self._buffer.clear()
        return self.policy if self.num_agents == 1 else self._policies


__all__ = ["PolicyTrainer"]

