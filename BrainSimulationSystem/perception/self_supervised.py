"""Self-supervised learners for perception modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional
from collections import deque

import numpy as np


@dataclass
class ContrastiveLearnerConfig:
    feature_dim: int
    projection_dim: int = 64
    buffer_size: int = 512
    batch_size: int = 64
    temperature: float = 0.5
    learning_rate: float = 1e-3


class ContrastiveLearner:
    """Simple SimCLR-style learner implemented purely with NumPy."""

    def __init__(self, config: ContrastiveLearnerConfig) -> None:
        self.config = config
        self.buffer: Deque[np.ndarray] = deque(maxlen=self.config.buffer_size)
        scale = 1.0 / np.sqrt(self.config.feature_dim)
        self.projection = np.random.randn(
            self.config.feature_dim, self.config.projection_dim
        ).astype(np.float32) * scale
        self._last_loss: float | None = None

    def observe(self, feature: np.ndarray) -> None:
        feature = np.asarray(feature, dtype=np.float32).reshape(-1)
        self.buffer.append(feature)
        if len(self.buffer) >= self.config.batch_size:
            self.train_step()

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < self.config.batch_size:
            return None
        indices = np.random.choice(len(self.buffer), size=self.config.batch_size, replace=False)
        batch = np.stack([self.buffer[i] for i in indices], axis=0)
        loss = self._simclr_update(batch)
        self._last_loss = loss
        return loss

    def _simclr_update(self, batch: np.ndarray) -> float:
        view1 = self._augment(batch)
        view2 = self._augment(batch)
        z1, y1 = self._project(view1)
        z2, y2 = self._project(view2)

        logits = (z1 @ z2.T) / self.config.temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        positives = np.diag(exp_logits)
        denom = exp_logits.sum(axis=1)
        loss = -np.mean(np.log(positives / denom + 1e-12))

        softmax = exp_logits / denom[:, None]
        grad_z1 = -(z2 / self.config.temperature) + (softmax @ z2) / self.config.temperature
        grad_z2 = -(z1 / self.config.temperature) + (softmax.T @ z1) / self.config.temperature

        grad_y1 = grad_z1 / (np.linalg.norm(y1, axis=1, keepdims=True) + 1e-8)
        grad_y2 = grad_z2 / (np.linalg.norm(y2, axis=1, keepdims=True) + 1e-8)

        grad_w = view1.T @ grad_y1 + view2.T @ grad_y2
        self.projection -= self.config.learning_rate * grad_w
        return float(loss)

    def _project(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = features @ self.projection
        z = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-8)
        return z, y

    def _augment(self, features: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0.0, 0.02, size=features.shape).astype(np.float32)
        scale = np.random.uniform(0.9, 1.1, size=(features.shape[0], 1)).astype(np.float32)
        return features * scale + noise

    @property
    def last_loss(self) -> Optional[float]:
        return self._last_loss
