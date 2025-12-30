from __future__ import annotations

"""Asynchronous Advantage Actor-Critic (A3C) implementation."""

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn, optim


@dataclass
class A3CConfig:
    lr: float = 1e-3
    gamma: float = 0.99


class A3C:
    """Simplified A3C trainer with configurable hyperparameters."""

    def __init__(self, policy: nn.Module, value: nn.Module, config: A3CConfig | None = None):
        self.policy = policy
        self.value = value
        self.config = config or A3CConfig()
        params = list(policy.parameters()) + list(value.parameters())
        self.optimizer = optim.Adam(params, lr=self.config.lr)

    def update(self, trajectories: Iterable[dict]) -> None:
        """Update networks from a batch of trajectories.

        Each trajectory is expected to be a dict with keys ``state``, ``action``,
        ``reward``, ``next_state`` and ``done`` containing tensors. This method
        performs a single synchronous update using the A3C objective. In a true
        A3C implementation the gradients would be accumulated asynchronously from
        multiple workers; here we simply combine trajectories.
        """

        for t in trajectories:
            s = t["state"]
            a = t["action"]
            r = t["reward"]
            ns = t["next_state"]
            done = t["done"]

            with torch.no_grad():
                target = r + self.config.gamma * self.value(ns) * (1 - done)
            value = self.value(s)
            advantage = target - value

            dist = self.policy(s)
            logp = dist.log_prob(a)
            policy_loss = -(logp * advantage.detach()).mean()
            value_loss = advantage.pow(2).mean()

            loss = policy_loss + value_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


__all__ = ["A3C", "A3CConfig"]
