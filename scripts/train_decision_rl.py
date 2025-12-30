#!/usr/bin/env python3
"""Train a lightweight RL policy for :class:`DecisionProcess`."""

from __future__ import annotations

import argparse
import os
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import gym
    _GYMNASIUM_STYLE = gym.__name__ == "gymnasium"
except Exception:  # pragma: no cover - fallback to gymnasium
    import gymnasium as gym  # type: ignore
    _GYMNASIUM_STYLE = True

from BrainSimulationSystem.config.default_config import get_config
from BrainSimulationSystem.decision import deep_rl_agent as rl_module
from BrainSimulationSystem.decision.deep_rl_agent import (
    DecisionRLAgent,
    RLAgentConfig,
    build_option_observation,
)
from BrainSimulationSystem.learning.experience import ExperienceLearningSystem
from BrainSimulationSystem.models.decision import DecisionProcess


class DecisionRankingEnv(gym.Env):
    """Environment presenting decision options sequentially for scoring."""

    metadata = {"render.modes": []}

    def __init__(
        self,
        decision_params: Dict[str, Any],
        seed: Optional[int] = None,
        max_options: int = 5,
    ) -> None:
        super().__init__()
        self.seed_value = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.rng = np.random.default_rng(self.seed_value)
        self.max_options = max(2, max_options)

        params = deepcopy(decision_params)
        params.setdefault("decision_type", "softmax")
        params.setdefault("rl", {})
        params["rl"]["enabled"] = False
        self.decision_model = DecisionProcess(None, params)
        self.experience = ExperienceLearningSystem()

        self.current_context: Dict[str, Any] = {}
        self.current_options: List[Dict[str, Any]] = []
        self.current_index: int = 0
        self.predicted_scores: List[float] = []
        self.expert_index: int = 0

        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._reset_episode()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):  # type: ignore[override]
        if seed is not None:
            self.seed_value = seed
            self.rng = np.random.default_rng(seed)

        obs = self._reset_episode()
        if _GYMNASIUM_STYLE:
            return obs, {}
        return obs

    def step(self, action):  # type: ignore[override]
        score = float(np.asarray(action).mean())
        self.predicted_scores.append(score)
        self.current_index += 1

        terminated = self.current_index >= len(self.current_options)
        reward = 0.0
        info: Dict[str, Any] = {}

        if terminated:
            best_idx = int(np.argmax(self.predicted_scores)) if self.predicted_scores else 0
            reward = 1.0 if best_idx == self.expert_index else -1.0
            obs = np.zeros(8, dtype=np.float32)
            state = {"context": self.current_context, "options": self.current_options}
            self.experience.store_experience(
                state,
                best_idx,
                reward,
                state,
                available_actions=list(range(len(self.current_options))),
            )
        else:
            obs = self._current_observation()

        if _GYMNASIUM_STYLE:
            return obs, reward, terminated, False, info
        return obs, reward, terminated, info

    def _reset_episode(self) -> np.ndarray:
        option_count = int(self.rng.integers(2, self.max_options + 1))
        self.current_context = self._sample_context()
        self.current_options = [self._sample_option() for _ in range(option_count)]
        self.predicted_scores = []
        self.current_index = 0

        expert_result = self.decision_model.process(
            {"options": self.current_options, "context": self.current_context}
        )
        expert_decision = expert_result.get("decision")
        self.expert_index = (
            self.current_options.index(expert_decision)
            if expert_decision in self.current_options
            else 0
        )

        return self._current_observation()

    def _current_observation(self) -> np.ndarray:
        option = self.current_options[self.current_index]
        return build_option_observation(
            self.current_context,
            option,
            self.current_index,
            len(self.current_options),
        )

    def _sample_context(self) -> Dict[str, float]:
        return {
            "urgency": float(self.rng.random()),
            "resource_level": float(self.rng.random()),
            "stress": float(self.rng.random()),
            "arousal": float(self.rng.random()),
        }

    def _sample_option(self) -> Dict[str, Any]:
        return {
            "expected_value": float(self.rng.uniform(0.0, 1.0)),
            "risk": float(self.rng.uniform(0.0, 1.0)),
            "cost": float(self.rng.uniform(0.0, 1.0)),
            "stress": float(self.rng.uniform(0.0, 1.0)),
            "arousal": float(self.rng.uniform(0.0, 1.0)),
            "social_norms": {
                "peer_support": float(self.rng.uniform(-1.0, 1.0)),
                "policy_alignment": float(self.rng.uniform(-1.0, 1.0)),
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="prototype", help="Configuration profile to load")
    parser.add_argument("--algorithm", default="ppo", choices=["ppo", "dqn"], help="RL algorithm")
    parser.add_argument("--timesteps", type=int, default=5000, help="Training timesteps per update")
    parser.add_argument("--max-options", type=int, default=5, help="Maximum options sampled per context")
    parser.add_argument("--model-path", default=None, help="Explicit path for the trained policy")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args()


def build_env_factory(decision_params: Dict[str, Any], max_options: int, seed: Optional[int]):
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)

    def _factory() -> gym.Env:
        return DecisionRankingEnv(decision_params, seed=seed, max_options=max_options)

    return _factory


def main() -> int:
    args = parse_args()

    if not rl_module._STABLE_BASELINES_AVAILABLE or gym is None:  # type: ignore[attr-defined]
        print("stable_baselines3 and gym/gymnasium are required for RL training.")
        return 1

    config = get_config(args.profile)
    decision_params = deepcopy(config.get("decision", {}))
    rl_config = decision_params.get("rl", {})

    model_path = args.model_path or rl_config.get("model_path")
    if not model_path:
        planner_cfg = config.get("planner", {}).get("controller", {})
        model_path = planner_cfg.get("rl_model_path", "BrainSimulationSystem/models/rl/decision_policy.zip")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    rl_agent_config = RLAgentConfig(
        algorithm=args.algorithm,
        policy=rl_config.get("policy", "MlpPolicy"),
        model_path=model_path,
        device=rl_config.get("device", "auto"),
        verbose=int(rl_config.get("verbose", 0)),
    )

    env_factory = build_env_factory(decision_params, args.max_options, args.seed)
    agent = DecisionRLAgent(config=rl_agent_config, env_builder=env_factory)

    result = agent.update(total_timesteps=args.timesteps, save_path=model_path)
    if result.get("status") != "updated":
        print(f"Training did not complete successfully: {result}")
        return 1

    agent.save(model_path)

    print(f"Saved RL policy to {model_path} using {args.algorithm.upper()} for {args.timesteps} timesteps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
