from __future__ import annotations

"""Minimal Proximal Policy Optimization implementation."""

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn, optim


@dataclass
class PPOConfig:
    lr: float = 3e-4
    clip_ratio: float = 0.2
    epochs: int = 4
    batch_size: int = 64


class PPO:
    """Simple PPO trainer with configurable hyperparameters."""

    def __init__(self, policy: nn.Module, value: nn.Module, config: PPOConfig | None = None):
        self.policy = policy
        self.value = value
        self.config = config or PPOConfig()
        params = list(policy.parameters()) + list(value.parameters())
        self.optimizer = optim.Adam(params, lr=self.config.lr)

    def update(self, trajectories: Iterable[dict]) -> None:
        """Update policy and value networks using collected trajectories.

        Each trajectory element is expected to be a dict containing tensors with
        keys ``state``, ``action``, ``logp``, ``return`` and ``adv``. This method
        performs a number of epochs of minibatch updates using the clipped PPO
        objective. The implementation is intentionally lightweight and intended
        for small experiments rather than production use.
        """

        data = {k: torch.cat([t[k] for t in trajectories]) for k in trajectories[0].keys()}  # type: ignore[index]
        n = data["state"].size(0)
        for _ in range(self.config.epochs):
            idx = torch.randperm(n)
            for start in range(0, n, self.config.batch_size):
                end = start + self.config.batch_size
                batch_idx = idx[start:end]
                s = data["state"][batch_idx]
                a = data["action"][batch_idx]
                logp_old = data["logp"][batch_idx]
                ret = data["return"][batch_idx]
                adv = data["adv"][batch_idx]

                dist = self.policy(s)
                logp = dist.log_prob(a)
                ratio = (logp - logp_old).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(self.value(s).squeeze(-1), ret)
                loss = policy_loss + value_loss * 0.5

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


__all__ = ["PPO", "PPOConfig"]
