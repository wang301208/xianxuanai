from __future__ import annotations

"""Experience replay buffers with optional prioritized sampling."""

from dataclasses import dataclass, field
from typing import List, Tuple
import random

Transition = Tuple


@dataclass
class ReplayBuffer:
    """Replay buffer supporting uniform or prioritized sampling.

    Prioritized sampling follows the algorithm described in
    *Prioritized Experience Replay* (Schaul et al., 2016). When
    ``prioritized`` is ``True`` the buffer maintains a parallel list of
    priorities which are used to weight samples. New transitions are given the
    maximum priority by default so that they are sampled at least once before
    being updated.
    """

    capacity: int
    prioritized: bool = False
    alpha: float = 0.6
    beta: float = 0.4
    _storage: List[Transition] = field(default_factory=list, init=False)
    _priorities: List[float] = field(default_factory=list, init=False)
    _pos: int = field(default=0, init=False)

    def push(self, *transition: Transition) -> None:
        """Add a transition to the buffer."""

        if len(self._storage) < self.capacity:
            self._storage.append(transition)
            if self.prioritized:
                self._priorities.append(max(self._priorities, default=1.0))
        else:
            self._storage[self._pos] = transition
            if self.prioritized:
                self._priorities[self._pos] = max(self._priorities, default=1.0)
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[Transition], List[int], List[float]]:
        """Sample a batch of transitions.

        Returns ``(batch, indices, weights)`` where ``indices`` are the indices
        into the underlying storage and ``weights`` are importance-sampling
        weights. For uniform sampling the weights are all ``1``.
        """

        if self.prioritized and self._priorities:
            probs = [p ** self.alpha for p in self._priorities]
            total = sum(probs)
            probs = [p / total for p in probs]
            indices = random.choices(range(len(self._storage)), probs, k=batch_size)
            weights = [(len(self._storage) * probs[i]) ** (-self.beta) for i in indices]
            max_w = max(weights, default=1.0)
            weights = [w / max_w for w in weights]
        else:
            indices = random.choices(range(len(self._storage)), k=batch_size)
            weights = [1.0] * batch_size
        batch = [self._storage[i] for i in indices]
        return batch, indices, weights

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Update sampling priorities for given indices."""

        if not self.prioritized:
            return
        for idx, prio in zip(indices, priorities):
            self._priorities[idx] = float(prio)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._storage)


__all__ = ["ReplayBuffer"]
