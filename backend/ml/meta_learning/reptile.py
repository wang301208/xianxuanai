from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .maml import TaskData


@dataclass
class Reptile:
    """Simple NumPy implementation of the Reptile meta-learning algorithm."""

    input_dim: int
    inner_lr: float = 0.01
    meta_lr: float = 0.1
    adapt_steps: int = 1

    def __post_init__(self) -> None:
        self.weights = np.zeros(self.input_dim)

    def _loss_and_grad(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        preds = X @ w
        diff = preds - y
        grad = 2 * X.T @ diff / len(X)
        return grad

    def adapt(self, task: TaskData) -> np.ndarray:
        """Adapt model weights to a single task."""
        w = self.weights.copy()
        for _ in range(self.adapt_steps):
            grad = self._loss_and_grad(w, task.support_x, task.support_y)
            w -= self.inner_lr * grad
        return w

    def meta_train(self, tasks: List[TaskData], epochs: int = 1) -> List[float]:
        """Meta-train across ``tasks`` for ``epochs`` iterations."""
        history: List[float] = []
        for _ in range(epochs):
            epoch_loss = 0.0
            for task in tasks:
                adapted = self.adapt(task)
                grad = adapted - self.weights
                self.weights += self.meta_lr * grad
                # simple loss: squared error on query set
                preds = task.query_x @ adapted
                epoch_loss += float(np.mean((preds - task.query_y) ** 2))
            history.append(epoch_loss / len(tasks))
        return history
