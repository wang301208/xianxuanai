from __future__ import annotations

"""Soft Actor-Critic (SAC) implementation."""

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn, optim


@dataclass
class SACConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2


class SAC:
    """Minimal SAC trainer with configurable hyperparameters."""

    def __init__(
        self,
        policy: nn.Module,
        q1: nn.Module,
        q2: nn.Module,
        target_q1: nn.Module,
        target_q2: nn.Module,
        config: SACConfig | None = None,
    ):
        self.policy = policy
        self.q1 = q1
        self.q2 = q2
        self.target_q1 = target_q1
        self.target_q2 = target_q2
        self.config = config or SACConfig()
        params = list(policy.parameters()) + list(q1.parameters()) + list(q2.parameters())
        self.optimizer = optim.Adam(params, lr=self.config.lr)

    def soft_update(self) -> None:
        for target, source in ((self.target_q1, self.q1), (self.target_q2, self.q2)):
            for tp, p in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1 - self.config.tau).add_(p.data, alpha=self.config.tau)

    def update(self, batch: dict) -> None:
        s = batch["state"]
        a = batch["action"]
        r = batch["reward"]
        ns = batch["next_state"]
        done = batch["done"]

        with torch.no_grad():
            ndist = self.policy(ns)
            na = ndist.rsample()
            logp = ndist.log_prob(na)
            q1_t = self.target_q1(torch.cat([ns, na], dim=-1))
            q2_t = self.target_q2(torch.cat([ns, na], dim=-1))
            min_q = torch.min(q1_t, q2_t) - self.config.alpha * logp
            target = r + self.config.gamma * (1 - done) * min_q

        q1_v = self.q1(torch.cat([s, a], dim=-1))
        q2_v = self.q2(torch.cat([s, a], dim=-1))
        q_loss = nn.functional.mse_loss(q1_v, target) + nn.functional.mse_loss(q2_v, target)

        dist = self.policy(s)
        sa = dist.rsample()
        logp = dist.log_prob(sa)
        q1_pi = self.q1(torch.cat([s, sa], dim=-1))
        q2_pi = self.q2(torch.cat([s, sa], dim=-1))
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.config.alpha * logp - min_q_pi).mean()

        loss = q_loss + policy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update()


__all__ = ["SAC", "SACConfig"]
