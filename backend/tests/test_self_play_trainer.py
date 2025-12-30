"""Tests for the self-play trainer coordination module."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("torch")

import torch

from backend.algorithms import evolution_engine
from backend.algorithms.self_play_trainer import SelfPlayTrainer, SelfPlayTrainerConfig
from modules.benchmarks.problems import Sphere


@dataclass
class ToyEnv:
    """Minimal single-agent environment with a discrete action space."""

    target: int = 1

    def __post_init__(self) -> None:
        self.observation_space = _DiscreteSpace(1)
        self.action_space = _DiscreteSpace(2)
        self._step_count = 0

    def reset(self, seed: int | None = None) -> tuple[list[float], dict]:  # pragma: no cover - deterministic
        self._step_count = 0
        torch.manual_seed(seed or 0)
        return [0.0], {}

    def step(self, action: Any) -> tuple[list[float], float, bool, dict]:
        if isinstance(action, tuple):
            action = action[0]
        reward = 1.0 if int(action) == self.target else 0.0
        self._step_count += 1
        done = self._step_count >= 1
        return [float(action)], reward, done, {}


@dataclass
class _DiscreteSpace:
    n: int
    shape: tuple[int, ...] = (1,)

    def sample(self) -> int:  # pragma: no cover - not used in deterministic env
        return 0


def _env_factory(_task=None) -> ToyEnv:
    return ToyEnv()


def test_self_play_trainer_runs_and_records_metrics():
    trainer = SelfPlayTrainer(
        _env_factory,
        SelfPlayTrainerConfig(algorithm="ppo", episodes=3, max_steps_per_episode=2, seed=1),
    )

    result = trainer.train()

    assert result.algorithm == "ppo"
    assert len(result.metrics) == 3
    rewards = [m.reward for m in result.metrics]
    assert all(0.0 <= value <= 1.0 for value in rewards)
    action = result.policy([0.0], deterministic=True)
    assert action in (0, 1)


def test_optimize_registers_self_play_specialist():
    problem = Sphere(dim=2, bound=2.0)

    trainer = SelfPlayTrainer(
        _env_factory,
        SelfPlayTrainerConfig(algorithm="ppo", episodes=2, max_steps_per_episode=1, seed=2),
    )

    outcome = evolution_engine.optimize(
        problem,
        seed=3,
        max_iters=3,
        task_capabilities=("global_optimum", problem.name),
        self_play=trainer,
        self_play_specialist_name="toy_specialist",
        return_details=True,
    )

    vector, value, iterations, elapsed, details = outcome
    assert len(vector) == problem.dim
    assert math.isfinite(value)
    assert iterations > 0
    assert elapsed >= 0.0
    assert details["self_play"] is not None
    assert details["specialist"].name == "toy_specialist"
    assert details["specialist"].capabilities >= {"self_play"}

