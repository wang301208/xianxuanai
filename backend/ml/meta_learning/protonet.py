from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .maml import TaskData


@dataclass
class PrototypicalNetwork:
    """Simple Prototypical Network for few-shot classification."""

    input_dim: int
    embedding_dim: int = 16

    def __post_init__(self) -> None:
        self.embedding = np.random.randn(self.input_dim, self.embedding_dim) * 0.01
        self.prototypes: Dict[int, np.ndarray] | None = None

    def _embed(self, x: np.ndarray) -> np.ndarray:
        return x @ self.embedding

    def fit(self, task: TaskData) -> None:
        """Compute class prototypes from support set."""
        embeds = self._embed(task.support_x)
        self.prototypes = {}
        for label in np.unique(task.support_y):
            mask = task.support_y == label
            self.prototypes[int(label)] = embeds[mask].mean(axis=0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.prototypes is None:
            raise RuntimeError("Model has not been fitted")
        embeds = self._embed(x)
        preds: List[int] = []
        for e in embeds:
            dists = {c: np.linalg.norm(e - p) for c, p in self.prototypes.items()}
            preds.append(min(dists, key=dists.get))
        return np.array(preds)

    def score(self, task: TaskData) -> float:
        """Return classification accuracy on a task's query set."""
        self.fit(task)
        preds = self.predict(task.query_x)
        return float(np.mean(preds == task.query_y))
