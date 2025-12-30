"""Lightweight deep reinforcement learning agents (DQN & PPO)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import random

import numpy as np

try:  # pragma: no cover - torch may not be available in minimal setups
    import torch
    from torch import nn
    from torch.nn import functional as F
except Exception:  # pragma: no cover - fallback when torch missing
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


TORCH_AVAILABLE = torch is not None


# --------------------------------------------------------------------------- #
# Shared utilities
# --------------------------------------------------------------------------- #


class ReplayBuffer:
    """Simple FIFO replay buffer for off-policy algorithms."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self._position = 0

    def __len__(self) -> int:
        return len(self._buffer)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        if len(self._buffer) < self.capacity:
            self._buffer.append((state, action, reward, next_state, done))
        else:
            self._buffer[self._position] = (state, action, reward, next_state, done)
        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self._buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return (
            torch.from_numpy(states).float(),
            torch.from_numpy(actions).long(),
            torch.from_numpy(rewards).float(),
            torch.from_numpy(next_states).float(),
            torch.from_numpy(dones.astype(np.float32)).float(),
        )


# --------------------------------------------------------------------------- #
# Deep Q-Network
# --------------------------------------------------------------------------- #


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000
    warmup_steps: int = 1_000
    update_interval: int = 4
    tau: float = 0.02
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995


if TORCH_AVAILABLE:

    class _QNetwork(nn.Module):
        def __init__(self, state_dim: int, action_dim: int) -> None:
            super().__init__()
            hidden = max(32, state_dim * 2)
            self.model = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, action_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

else:  # pragma: no cover - exercised when torch unavailable

    class _QNetwork:  # type: ignore[too-few-public-methods]
        pass


class DQNAgent:
    """Minimal DQN agent supporting epsilon-greedy exploration."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: DQNConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to use DQNAgent.")

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.config = config or DQNConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = _QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = _QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.config.lr)
        self.buffer = ReplayBuffer(self.config.buffer_size)
        self.steps = 0
        self.epsilon = self.config.epsilon_start

    # ------------------------------------------------------------------ #
    def select_action(self, state: np.ndarray) -> Tuple[int, Dict[str, float]]:
        self.steps += 1
        if random.random() < self.epsilon:
            action = random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                q_values = self.policy_net(tensor)
                action = int(torch.argmax(q_values, dim=1).item())
        self._decay_epsilon()
        return action, {"epsilon": self.epsilon}

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> Dict[str, float] | None:
        if (
            len(self.buffer) < max(self.config.batch_size, self.config.warmup_steps)
            or self.steps % self.config.update_interval != 0
        ):
            return None

        batch = self.buffer.sample(self.config.batch_size)
        states, actions, rewards, next_states, dones = batch
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target = rewards + self.config.gamma * (1.0 - dones) * next_q

        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self._soft_update()
        return {"loss": float(loss.item()), "buffer": float(len(self.buffer))}

    # ------------------------------------------------------------------ #
    def _decay_epsilon(self) -> None:
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay,
        )

    def _soft_update(self) -> None:
        tau = self.config.tau
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)


# --------------------------------------------------------------------------- #
# Proximal Policy Optimisation (on-policy)
# --------------------------------------------------------------------------- #


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_param: float = 0.2
    lr: float = 3e-4
    epochs: int = 4
    batch_size: int = 64


class RolloutBuffer:
    def __init__(self) -> None:
        self.reset()

    def add(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns(self, gamma: float, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
        returns: List[float] = []
        advantages: List[float] = []
        gae = 0.0
        next_value = 0.0

        for reward, value, done in zip(
            reversed(self.rewards), reversed(self.values), reversed(self.dones)
        ):
            delta = reward + gamma * next_value * (1.0 - float(done)) - value
            gae = delta + gamma * lam * (1.0 - float(done)) * gae
            advantages.insert(0, gae)
            next_value = value
        for advantage, value in zip(advantages, self.values):
            returns.append(advantage + value)

        return (
            torch.tensor(returns, dtype=torch.float32),
            torch.tensor(advantages, dtype=torch.float32),
        )

    def reset(self) -> None:
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []


if TORCH_AVAILABLE:

    class PPOAgent(nn.Module):
        """Tiny PPO implementation for discrete action spaces."""

        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            config: PPOConfig | None = None,
            device: torch.device | None = None,
        ) -> None:
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.config = config or PPOConfig()
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

            hidden = max(64, state_dim * 2)
            self.backbone = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
            )
            self.policy_head = nn.Linear(hidden, action_dim)
            self.value_head = nn.Linear(hidden, 1)
            self.to(self.device)
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
            self.buffer = RolloutBuffer()

        # ------------------------------------------------------------------ #
        def select_action(self, state: np.ndarray) -> Tuple[int, Dict[str, float]]:
            tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            logits = self.policy_head(self.backbone(tensor))
            probs = torch.distributions.Categorical(logits=logits)
            action = probs.sample()
            value = self.value_head(self.backbone(tensor)).squeeze(-1)
            self.buffer.add(
                state,
                int(action.item()),
                float(probs.log_prob(action).item()),
                0.0,
                float(value.item()),
                False,
            )
            return int(action.item()), {"log_prob": float(probs.log_prob(action).item())}

        def observe(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            done: bool,
        ) -> None:
            if self.buffer.rewards:
                self.buffer.rewards[-1] = reward
                self.buffer.dones[-1] = done

        def update(self) -> Dict[str, float] | None:
            if not self.buffer.states:
                return None

            returns, advantages = self.buffer.compute_returns(self.config.gamma, self.config.lam)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            states = torch.tensor(np.stack(self.buffer.states), dtype=torch.float32).to(self.device)
            actions = torch.tensor(self.buffer.actions, dtype=torch.long).to(self.device)
            old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32).to(self.device)
            returns = returns.to(self.device)
            advantages = advantages.to(self.device)

            total_loss = 0.0
            for _ in range(self.config.epochs):
                logits = self.policy_head(self.backbone(states))
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions)
                ratios = torch.exp(log_probs - old_log_probs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - self.config.clip_param, 1.0 + self.config.clip_param) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                values = self.value_head(self.backbone(states)).squeeze(-1)
                value_loss = F.mse_loss(values, returns)
                entropy = dist.entropy().mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                self.optimizer.step()
                total_loss += float(loss.item())

            self.buffer.reset()
            return {"loss": total_loss / self.config.epochs}

else:  # pragma: no cover - fallback when torch unavailable

    class PPOAgent:  # type: ignore[too-few-public-methods]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("PyTorch is required to use PPOAgent.")


__all__ = [
    "DQNAgent",
    "DQNConfig",
    "PPOAgent",
    "PPOConfig",
    "TORCH_AVAILABLE",
]
