from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .maml import MAML, TaskData
from .protonet import PrototypicalNetwork
from .reptile import Reptile


@dataclass
class MetaLearningTrainer:
    """Unified interface for multiple meta-learning algorithms.

    Parameters
    ----------
    algorithm:
        Name of the algorithm to use (``"maml"``, ``"reptile"``, or ``"protonet"``).
    input_dim:
        Dimensionality of the input features.
    inner_lr, meta_lr, adapt_steps, embedding_dim:
        Hyperparameters forwarded to the underlying implementation.
    """

    algorithm: str
    input_dim: int
    inner_lr: float = 0.01
    meta_lr: float = 0.001
    adapt_steps: int = 1
    embedding_dim: int = 16

    def __post_init__(self) -> None:
        algo = self.algorithm.lower()
        if algo == "maml":
            self.model = MAML(
                input_dim=self.input_dim,
                inner_lr=self.inner_lr,
                meta_lr=self.meta_lr,
                adapt_steps=self.adapt_steps,
            )
            self.metric = "loss"
        elif algo == "reptile":
            self.model = Reptile(
                input_dim=self.input_dim,
                inner_lr=self.inner_lr,
                meta_lr=self.meta_lr,
                adapt_steps=self.adapt_steps,
            )
            self.metric = "loss"
        elif algo in {"protonet", "prototypical"}:
            self.model = PrototypicalNetwork(
                input_dim=self.input_dim, embedding_dim=self.embedding_dim
            )
            self.metric = "accuracy"
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def train(self, tasks: List[TaskData], epochs: int = 1) -> List[float]:
        """Train or evaluate the selected algorithm over ``tasks``.

        For gradient-based methods (MAML/Reptile), this calls ``meta_train``. For
        Prototypical Networks, this computes classification accuracy for each task
        as no further meta-optimization is required.
        """

        if isinstance(self.model, PrototypicalNetwork):
            history: List[float] = []
            for _ in range(epochs):
                scores = [self.model.score(t) for t in tasks]
                history.append(float(np.mean(scores)))
            return history
        return self.model.meta_train(tasks, epochs=epochs)
