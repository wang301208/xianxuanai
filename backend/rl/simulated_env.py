"""Lightweight simulated task environments for reinforcement learning."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml


Vector2 = Tuple[int, int]


@dataclass
class SubTask:
    """Intermediate goal within a task."""

    name: str
    target: Vector2
    reward: float = 0.2
    action: str = "interact"
    tolerance: int = 0


@dataclass
class TaskConfig:
    """Configuration describing a simulated task."""

    name: str
    grid_size: Vector2
    start: Vector2
    goal: Vector2
    obstacles: List[Vector2] = field(default_factory=list)
    subtasks: List[SubTask] = field(default_factory=list)
    action_set: List[str] = field(
        default_factory=lambda: ["up", "down", "left", "right", "interact", "wait"]
    )
    max_steps: int = 40
    step_cost: float = -0.01
    collision_cost: float = -0.05
    goal_reward: float = 1.0
    drift: float = 0.0
    plan: List[str] = field(default_factory=list)


@dataclass
class TaskSuite:
    """Collection of simulated tasks."""

    tasks: Dict[str, TaskConfig]

    @property
    def action_space(self) -> List[str]:
        actions: set[str] = set()
        for task in self.tasks.values():
            actions.update(task.action_set)
        return sorted(actions)

    def sample(self, rng: Optional[random.Random] = None, name: Optional[str] = None) -> TaskConfig:
        if name:
            return self.tasks[name]
        rng = rng or random
        return rng.choice(list(self.tasks.values()))

    def __len__(self) -> int:
        return len(self.tasks)


def _vector_from(value: Sequence[int]) -> Vector2:
    return int(value[0]), int(value[1])


def load_task_suite(path: str | bytes) -> TaskSuite:
    """Load task suite definition from YAML."""

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tasks_cfg = config.get("tasks", {})
    tasks: Dict[str, TaskConfig] = {}
    for name, spec in tasks_cfg.items():
        subtasks = [
            SubTask(
                name=sub["name"],
                target=_vector_from(sub["target"]),
                reward=float(sub.get("reward", 0.2)),
                action=sub.get("action", "interact"),
                tolerance=int(sub.get("tolerance", 0)),
            )
            for sub in spec.get("subtasks", [])
        ]
        tasks[name] = TaskConfig(
            name=name,
            grid_size=_vector_from(spec["grid_size"]),
            start=_vector_from(spec["start"]),
            goal=_vector_from(spec["goal"]),
            obstacles=[_vector_from(ob) for ob in spec.get("obstacles", [])],
            subtasks=subtasks,
            action_set=list(spec.get("action_set", ["up", "down", "left", "right", "interact", "wait"])),
            max_steps=int(spec.get("max_steps", 40)),
            step_cost=float(spec.get("step_cost", -0.01)),
            collision_cost=float(spec.get("collision_cost", -0.05)),
            goal_reward=float(spec.get("goal_reward", 1.0)),
            drift=float(spec.get("drift", 0.0)),
            plan=list(spec.get("plan", [])),
        )
    return TaskSuite(tasks=tasks)


class SimulatedTaskEnv:
    """Simple grid-based task simulator with subgoal progression."""

    def __init__(self, task: TaskConfig, *, rng: Optional[random.Random] = None) -> None:
        self.task = task
        self.rng = rng or random.Random()
        self.position: Vector2 = task.start
        self.steps = 0
        self.success = False
        self.subtask_index = 0
        self.plan = task.plan or [sub.name for sub in task.subtasks] or ["execute_goal"]

    def reset(self, plan: Optional[Sequence[str]] = None) -> Dict[str, float]:
        self.position = self.task.start
        self.steps = 0
        self.success = False
        self.subtask_index = 0
        if plan:
            self.plan = list(plan)
        elif self.task.plan:
            self.plan = list(self.task.plan)
        else:
            self.plan = [sub.name for sub in self.task.subtasks] or ["execute_goal"]
        return self._observation()

    def step(self, action: str) -> Tuple[Dict[str, float], float, bool, Dict[str, float]]:
        self.steps += 1
        reward = self.task.step_cost
        info: Dict[str, float] = {}
        nxt = self._apply_action(action)
        if nxt in self.task.obstacles:
            reward += self.task.collision_cost
            info["collision"] = 1.0
        else:
            self.position = nxt

        if action == "interact":
            reward += self._handle_interaction(info)

        if self.position == self.task.goal and self.subtask_index >= len(self.task.subtasks):
            reward += self.task.goal_reward
            self.success = True

        done = self.success or self.steps >= self.task.max_steps
        obs = self._observation()
        if done:
            info["success"] = 1.0 if self.success else 0.0
        return obs, reward, done, info

    @property
    def max_steps(self) -> int:
        return self.task.max_steps

    def _apply_action(self, action: str) -> Vector2:
        x, y = self.position
        if self.task.drift > 0 and self.rng.random() < self.task.drift:
            action = self.rng.choice(self.task.action_set)
        if action == "up":
            y = max(0, y - 1)
        elif action == "down":
            y = min(self.task.grid_size[1] - 1, y + 1)
        elif action == "left":
            x = max(0, x - 1)
        elif action == "right":
            x = min(self.task.grid_size[0] - 1, x + 1)
        return (x, y)

    def _handle_interaction(self, info: Dict[str, float]) -> float:
        if self.subtask_index >= len(self.task.subtasks):
            return 0.0
        subtask = self.task.subtasks[self.subtask_index]
        if self._within_tolerance(self.position, subtask.target, subtask.tolerance):
            self.subtask_index += 1
            info[f"subtask:{subtask.name}"] = 1.0
            return subtask.reward
        return self.task.collision_cost * 0.5

    @staticmethod
    def _within_tolerance(pos: Vector2, target: Vector2, tolerance: int) -> bool:
        return abs(pos[0] - target[0]) <= tolerance and abs(pos[1] - target[1]) <= tolerance

    def _observation(self) -> Dict[str, float]:
        current_target = (
            self.task.subtasks[self.subtask_index].target
            if self.subtask_index < len(self.task.subtasks)
            else self.task.goal
        )
        progress = self.subtask_index / max(1, len(self.task.subtasks))
        return {
            "position_x": float(self.position[0]),
            "position_y": float(self.position[1]),
            "goal_x": float(self.task.goal[0]),
            "goal_y": float(self.task.goal[1]),
            "target_x": float(current_target[0]),
            "target_y": float(current_target[1]),
            "progress": progress,
            "steps_remaining": float(self.task.max_steps - self.steps),
            "subtask_index": float(self.subtask_index),
            "plan_length": float(len(self.plan)),
        }

    def observation_vector(self, observation: Optional[Dict[str, float]] = None) -> np.ndarray:
        obs = observation or self._observation()
        width = max(1, self.task.grid_size[0] - 1)
        height = max(1, self.task.grid_size[1] - 1)
        steps_norm = obs["steps_remaining"] / max(1.0, float(self.task.max_steps))
        vec = np.array(
            [
                obs["position_x"] / width,
                obs["position_y"] / height,
                obs["goal_x"] / width,
                obs["goal_y"] / height,
                (obs["target_x"] - obs["position_x"]) / max(1.0, width),
                (obs["target_y"] - obs["position_y"]) / max(1.0, height),
                obs["progress"],
                steps_norm,
                obs["subtask_index"] / max(1.0, obs["plan_length"]),
                obs["plan_length"] / 10.0,
            ],
            dtype=np.float32,
        )
        return vec

    def action_mask(self, observation: Optional[Dict[str, float]] = None) -> np.ndarray:
        obs = observation or self._observation()
        mask = np.ones(len(self.task.action_set), dtype=np.float32)
        if "interact" in self.task.action_set:
            interact_idx = self.task.action_set.index("interact")
            target = (
                self.task.subtasks[self.subtask_index].target
                if self.subtask_index < len(self.task.subtasks)
                else self.task.goal
            )
            allowed = self._within_tolerance(
                (int(obs["position_x"]), int(obs["position_y"])),
                target,
                0,
            )
            if not allowed:
                mask[interact_idx] = 0.0
        return mask


def merge_action_masks(
    mask: np.ndarray,
    task_actions: Sequence[str],
    global_actions: Sequence[str],
) -> np.ndarray:
    """Align per-task masks with the global action ordering."""

    aligned = np.ones(len(global_actions), dtype=np.float32)
    lookup = {action: i for i, action in enumerate(task_actions)}
    for idx, action in enumerate(global_actions):
        if action in lookup:
            aligned[idx] = mask[lookup[action]]
        else:
            aligned[idx] = 0.0
    return aligned


def vectorize_observation(
    env: SimulatedTaskEnv,
    observation: Dict[str, float],
    global_actions: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return feature vector and ready-to-use action mask."""

    features = env.observation_vector(observation)
    local_mask = env.action_mask(observation)
    mask = merge_action_masks(local_mask, env.task.action_set, global_actions)
    return features, mask
