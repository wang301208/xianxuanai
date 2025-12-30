"""Dynamic neural network with adaptive structure.

This module implements a simple feed-forward network that can grow or shrink
its hidden layers based on training feedback. The architecture adapts when the
training improvement falls below user-defined thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import yaml

from ..base import Algorithm


@dataclass
class DynamicConfig:
    """Configuration for :class:`DynamicNetwork` structure adaptation.

    Parameters
    ----------
    max_layers:
        Maximum number of hidden layers allowed in the network.
    growth_trigger:
        Minimum improvement required to avoid adding a new layer. If the loss
        improvement is below this threshold for ``patience`` consecutive
        epochs, a new layer is added.
    shrink_trigger:
        If the loss worsens beyond this negative threshold for ``patience``
        consecutive epochs, the most recent hidden layer is removed.
    patience:
        Number of consecutive epochs the trigger condition must persist before
        the structure is modified.
    layer_size:
        Number of units for each hidden layer.
    """

    max_layers: int = 5
    growth_trigger: float = 0.01
    shrink_trigger: float = -0.1
    patience: int = 1
    layer_size: int = 16


class DynamicNetwork(Algorithm):
    """Feed-forward network supporting dynamic layer growth and pruning."""

    def __init__(self, input_size: int, output_size: int, config: DynamicConfig | None = None) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.config = config or DynamicConfig()

        # Initialize with a single hidden layer.
        first = self._init_layer(input_size, self.config.layer_size)
        self.layers: List[dict[str, np.ndarray]] = [first]
        self.output_layer = self._init_layer(self.config.layer_size, output_size)

        self.prev_loss: float | None = None
        self.growth_counter = 0
        self.shrink_counter = 0

    # Layer helpers -------------------------------------------------
    def _init_layer(self, in_size: int, out_size: int) -> dict[str, np.ndarray]:
        return {
            "weights": np.random.randn(in_size, out_size) * 0.1,
            "bias": np.zeros(out_size),
        }

    def add_layer(self) -> None:
        last_out = self.layers[-1]["weights"].shape[1]
        self.layers.append(self._init_layer(last_out, self.config.layer_size))
        self.output_layer = self._init_layer(self.config.layer_size, self.output_size)

    def remove_layer(self) -> None:
        if len(self.layers) > 1:
            self.layers.pop()
            last_out = self.layers[-1]["weights"].shape[1]
            self.output_layer = self._init_layer(last_out, self.output_size)

    # Forward pass --------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = np.tanh(x @ layer["weights"] + layer["bias"])
        return x @ self.output_layer["weights"] + self.output_layer["bias"]

    # Adaptation ----------------------------------------------------
    def adapt_structure(self, improvement: float) -> None:
        cfg = self.config
        if improvement < cfg.shrink_trigger and len(self.layers) > 1:
            self.shrink_counter += 1
            self.growth_counter = 0
        elif improvement < cfg.growth_trigger and len(self.layers) < cfg.max_layers:
            self.growth_counter += 1
            self.shrink_counter = 0
        else:
            self.growth_counter = 0
            self.shrink_counter = 0

        if self.shrink_counter >= cfg.patience:
            self.remove_layer()
            self.shrink_counter = 0
        elif self.growth_counter >= cfg.patience:
            self.add_layer()
            self.growth_counter = 0

    # Training ------------------------------------------------------
    def execute(self, data: np.ndarray, targets: np.ndarray, epochs: int = 1, lr: float = 0.01) -> np.ndarray:
        """Train the network on ``data`` for a number of epochs.

        This minimal implementation performs gradient-free updates and focuses on
        demonstrating structure adaptation rather than accurate learning.
        """

        for _ in range(epochs):
            preds = self.forward(data)
            loss = float(np.mean((preds - targets) ** 2))
            improvement = float("inf") if self.prev_loss is None else self.prev_loss - loss
            self.adapt_structure(improvement)
            self.prev_loss = loss
        return preds


def load_config(path: str) -> DynamicConfig:
    """Load :class:`DynamicConfig` from a YAML file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return DynamicConfig(**data)
