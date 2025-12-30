"""Simple simulated environments to close the perception-action loop."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np


class BaseEnvironment:
    """Abstract environment interface."""

    def reset(self) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    def step(self, action: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        raise NotImplementedError

    def render(self) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class GridWorldEnvironment(BaseEnvironment):
    """Lightweight 2-D grid environment providing visual/audio observations."""

    width: int = 8
    height: int = 8
    max_steps: int = 64
    position: Tuple[int, int] = field(init=False, default=(0, 0))
    goal: Tuple[int, int] = field(init=False, default=(7, 7))
    steps: int = field(init=False, default=0)

    def reset(self) -> Dict[str, Any]:
        self.position = (0, 0)
        self.goal = (self.width - 1, self.height - 1)
        self.steps = 0
        return self._build_observation(reward=0.0, done=False, info={"event": "reset"})

    def step(self, action: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        params = params or {}
        dx, dy = 0, 0
        action_lower = action.lower()
        if action_lower in {"move_north", "up"}:
            dy = -1
        elif action_lower in {"move_south", "down"}:
            dy = 1
        elif action_lower in {"move_west", "left"}:
            dx = -1
        elif action_lower in {"move_east", "right"}:
            dx = 1
        elif action_lower == "move":
            dx = int(params.get("dx", 0))
            dy = int(params.get("dy", 0))

        x = int(np.clip(self.position[0] + dx, 0, self.width - 1))
        y = int(np.clip(self.position[1] + dy, 0, self.height - 1))
        self.position = (x, y)
        self.steps += 1

        distance = math.hypot(self.goal[0] - x, self.goal[1] - y)
        reward = -0.01 * self.steps
        done = self.steps >= self.max_steps or self.position == self.goal
        if self.position == self.goal:
            reward += 1.0

        info = {
            "position": self.position,
            "goal": self.goal,
            "distance": distance,
            "steps": self.steps,
            "action": action_lower,
        }
        return self._build_observation(reward=reward, done=done, info=info)

    def render(self) -> Dict[str, Any]:
        return self._build_observation(reward=0.0, done=False, info={"event": "render"})

    # ------------------------------------------------------------------
    def _build_observation(self, *, reward: float, done: bool, info: Dict[str, Any]) -> Dict[str, Any]:
        info = dict(info)
        if "distance" not in info:
            info["distance"] = math.hypot(
                self.goal[0] - self.position[0],
                self.goal[1] - self.position[1],
            )
        vision = np.zeros((32, 32), dtype=np.float32)
        goal_x = int((self.goal[0] / max(1, self.width - 1)) * (vision.shape[1] - 1))
        goal_y = int((self.goal[1] / max(1, self.height - 1)) * (vision.shape[0] - 1))
        pos_x = int((self.position[0] / max(1, self.width - 1)) * (vision.shape[1] - 1))
        pos_y = int((self.position[1] / max(1, self.height - 1)) * (vision.shape[0] - 1))
        vision[goal_y : goal_y + 2, goal_x : goal_x + 2] = 0.8
        vision[pos_y : pos_y + 2, pos_x : pos_x + 2] = 1.0
        vision = np.clip(vision, 0.0, 1.0)

        audio_length = 64
        t = np.linspace(0, 1, audio_length, dtype=np.float32)
        frequency = 220.0 + 10.0 * info["distance"]
        audio = np.sin(2 * np.pi * frequency * t) * np.exp(-0.5 * t * self.steps)

        observation = {
            "vision": vision.tolist(),
            "audio": audio.tolist(),
        }
        return {
            "observation": observation,
            "reward": float(reward),
            "done": bool(done),
            "info": info,
        }
