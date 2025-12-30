from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class PolicyConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 128
    num_layers: int = 2


class AutoGPTPolicyHead(nn.Module):
    """Feed-forward policy head producing categorical distributions."""

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        self.network = self._build_network(config)

    @staticmethod
    def _build_network(config: PolicyConfig) -> nn.Sequential:
        hidden_dim = max(1, int(config.hidden_dim))
        layers = max(1, int(getattr(config, "num_layers", 2)))

        modules: list[nn.Module] = [
            nn.Linear(int(config.state_dim), hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(max(0, layers - 1)):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden_dim, int(config.action_dim)))
        return nn.Sequential(*modules)

    def update_architecture(self, *, hidden_dim: int | None = None, num_layers: int | None = None) -> bool:
        next_hidden = self.config.hidden_dim if hidden_dim is None else int(hidden_dim)
        next_layers = getattr(self.config, "num_layers", 2) if num_layers is None else int(num_layers)
        next_cfg = PolicyConfig(
            state_dim=int(self.config.state_dim),
            action_dim=int(self.config.action_dim),
            hidden_dim=int(next_hidden),
            num_layers=int(next_layers),
        )
        if next_cfg == self.config:
            return False
        self.config = next_cfg
        self.network = self._build_network(next_cfg)
        return True

    def forward(self, state: Tensor) -> torch.distributions.Categorical:
        if state.dim() == 1:
            logits = self.network(state)
        else:
            logits = self.network(state)
        return torch.distributions.Categorical(logits=logits)
