"""Simple RL-style optimizer for self-improvement actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random


@dataclass
class ActionValue:
    action: str
    value: float


class SelfOptimizationAgent:
    """Epsilon-greedy action selector with tabular value updates."""

    def __init__(self, actions: List[str], epsilon: float = 0.2, lr: float = 0.1, discount: float = 0.95) -> None:
        self.actions = actions
        self.epsilon = epsilon
        self.lr = lr
        self.discount = discount
        self.q_values: Dict[str, float] = {action: 0.0 for action in actions}
        self.prev_action: str | None = None

    def select_action(self) -> str:
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            action = max(self.actions, key=lambda a: self.q_values.get(a, 0.0))
        self.prev_action = action
        return action

    def update(self, reward: float) -> ActionValue:
        action = self.prev_action
        if action is None:
            return ActionValue(action="", value=0.0)
        old = self.q_values.get(action, 0.0)
        new = old + self.lr * (reward + self.discount * max(self.q_values.values()) - old)
        self.q_values[action] = new
        return ActionValue(action=action, value=new)
