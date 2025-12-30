"""Actor-critic reinforcement learning agent utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
from torch import nn
from torch.distributions import Categorical


@dataclass
class RLTrainingConfig:
    """Hyper-parameters controlling RL training."""

    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5


@dataclass
class TrainingStats:
    """Aggregate metrics produced during training."""

    episodes: int
    mean_return: float
    mean_length: float
    success_rate: float
    eval_return: float
    eval_success_rate: float


class ActorCriticAgent(nn.Module):
    """Simple actor-critic agent for discrete action spaces."""

    def __init__(self, obs_dim: int, action_dim: int, config: RLTrainingConfig | None = None) -> None:
        super().__init__()
        self.config = config or RLTrainingConfig()
        hidden = 128
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head = nn.Linear(hidden, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value

    def act(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = obs.to(self.device)
        logits, value = self.forward(obs)
        if mask is not None:
            mask = mask.to(self.device)
            invalid = mask < 0.5
            logits = logits.masked_fill(invalid, -1e9)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(dist.probs)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return int(action.item()), log_prob, value.squeeze(-1), entropy

    def compute_loss(
        self, transitions: Sequence[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute actor-critic loss components without applying an optimizer step."""

        if not transitions:
            zero = torch.zeros((), device=self.device, requires_grad=True)
            return zero, zero, zero, zero

        rewards = []
        log_probs: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        returns: List[float] = []
        R = 0.0

        for log_prob, value, reward, entropy in reversed(transitions):
            R = reward + self.config.gamma * R
            returns.insert(0, R)
            log_probs.insert(0, log_prob)
            values.insert(0, value)
            entropies.insert(0, entropy)

        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        log_probs_tensor = torch.stack(log_probs).to(self.device)
        values_tensor = torch.stack(values).to(self.device)
        entropies_tensor = torch.stack(entropies).to(self.device)

        advantages = returns_tensor - values_tensor.detach()
        policy_loss = -(log_probs_tensor * advantages).mean()
        value_loss = advantages.pow(2).mean()
        entropy_loss = entropies_tensor.mean()

        loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_loss

        return loss, policy_loss, value_loss, entropy_loss

    def update(
        self, transitions: Sequence[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]]
    ) -> Tuple[float, float, float]:
        if not transitions:
            return 0.0, 0.0, 0.0

        loss, policy_loss, value_loss, _entropy_loss = self.compute_loss(transitions)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return (
            float(loss.item()),
            float(policy_loss.item()),
            float(value_loss.item()),
        )
