"""Reinforcement learning agent for resource allocation using DQN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
try:  # pragma: no cover - optional torch dependency
    import torch  # type: ignore
    from torch import nn, optim  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = optim = None  # type: ignore


if nn is not None:
    class _DQN(nn.Module):
        def __init__(self, state_dim: int = 2, action_dim: int = 3) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 16),
                nn.ReLU(),
                nn.Linear(16, action_dim),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
            return self.net(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float


class ResourceRL:
    """Minimal DQN agent for resource optimisation with optional GPU support."""

    def __init__(self, gamma: float = 0.9, device: object | None = None) -> None:
        self.gamma = gamma
        self._torch_available = torch is not None and nn is not None and optim is not None
        self.device = None
        self.policy = None
        self.optimizer = None
        if self._torch_available:
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.policy = _DQN().to(self.device)
            self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01)

    # ------------------------------------------------------------------
    def select_action(self, cpu: float, mem: float) -> int:
        """Choose an action based on current predictions."""

        if not self._torch_available or self.policy is None or torch is None:
            if cpu > 80 or mem > 80:
                return 0
            if cpu < 20 and mem < 20:
                return 2
            return 1

        state = torch.tensor([[cpu, mem]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_vals = self.policy(state)
        return int(torch.argmax(q_vals).item())

    # ------------------------------------------------------------------
    def train(self, transitions: Iterable[Transition], epochs: int = 50) -> None:
        """Train the DQN from a batch of transitions."""

        if not self._torch_available or self.policy is None or torch is None:
            return
        batch = list(transitions)
        if not batch:
            return

        states_np = np.array([t.state for t in batch], dtype=np.float32)
        states = torch.from_numpy(states_np).to(self.device)
        targets = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)

        for _ in range(epochs):
            q_vals = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze()
            loss = nn.functional.mse_loss(q_vals, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # ------------------------------------------------------------------
    def evaluate(self, samples: Iterable[Tuple[float, float]]) -> float:
        """Return average reward for given samples."""

        arr = np.array(list(samples), dtype=np.float32)
        if arr.size == 0:
            return 0.0

        if not self._torch_available or self.policy is None or torch is None:
            actions = np.array([self.select_action(float(c), float(m)) for c, m in arr], dtype=np.int64)
        else:
            states = torch.tensor(arr, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                actions = torch.argmax(self.policy(states), dim=1).cpu().numpy()

        cpu_vals = arr[:, 0]
        mem_vals = arr[:, 1]
        rewards = np.where(
            (cpu_vals > 80) | (mem_vals > 80),
            np.where(actions == 0, 1.0, -1.0),
            np.where(
                (cpu_vals < 20) & (mem_vals < 20),
                np.where(actions == 2, 1.0, -1.0),
                np.where(actions == 1, 1.0, -1.0),
            ),
        )
        return float(rewards.mean())

    # ------------------------------------------------------------------
    @staticmethod
    def _rule_reward(state: np.ndarray, action: int) -> float:
        """Simple rule-based reward used for training/evaluation.

        Action mapping: 0=scale down, 1=keep, 2=scale up.
        Reward is +1 for correct decision according to thresholds, else -1.
        """

        cpu, mem = state
        if cpu > 80 or mem > 80:
            return 1.0 if action == 0 else -1.0
        if cpu < 20 and mem < 20:
            return 1.0 if action == 2 else -1.0
        return 1.0 if action == 1 else -1.0


if __name__ == "__main__":  # pragma: no cover - manual demo
    # Simple training and evaluation script
    data = [np.array([10.0, 10.0]), np.array([90.0, 90.0]), np.array([50.0, 50.0])]
    agent = ResourceRL()
    transitions = [
        Transition(state, action=0, reward=agent._rule_reward(state, 0)) for state in data
    ]
    agent.train(transitions)
    print("Average reward", agent.evaluate((s for s in data)))
