#!/usr/bin/env python3
"""Train a PPO policy that scores HybridPlanner candidate actions."""

from __future__ import annotations

import argparse
import os
import random
from copy import deepcopy
from typing import Any, Dict, Optional

from BrainSimulationSystem.config.default_config import get_config
from BrainSimulationSystem.decision import deep_rl_agent as rl_module
from BrainSimulationSystem.decision.deep_rl_agent import DecisionRLAgent, RLAgentConfig
from BrainSimulationSystem.planning.rl_env import PlannerRankingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="prototype", help="Configuration profile to load")
    parser.add_argument("--policy", default="MlpPolicy", help="Stable-Baselines policy class")
    parser.add_argument("--timesteps", type=int, default=10_000, help="Training timesteps per update")
    parser.add_argument("--max-options", type=int, default=6, help="Maximum options sampled per context")
    parser.add_argument("--model-path", default=None, help="Path where the trained policy will be saved")
    parser.add_argument("--device", default="auto", help="Computation device passed to Stable-Baselines3")
    parser.add_argument("--verbose", type=int, default=0, help="Stable-Baselines verbosity level")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args()


def build_env_factory(
    planner_config: Dict[str, Any],
    max_options: int,
    seed: Optional[int],
):
    base_seed = seed if seed is not None else random.randint(0, 2**32 - 1)

    def _factory() -> PlannerRankingEnv:
        # Offset the seed slightly to decorrelate vectorised environments
        env_seed = base_seed + random.randint(0, 10_000)
        return PlannerRankingEnv(planner_config, seed=env_seed, max_options=max_options)

    return _factory


def main() -> int:
    args = parse_args()

    if not rl_module._STABLE_BASELINES_AVAILABLE:  # type: ignore[attr-defined]
        print("stable_baselines3 and gym/gymnasium are required for planner RL training.")
        return 1

    config = get_config(args.profile)
    planner_cfg = deepcopy(config.get("planner", {}).get("controller", {}))

    model_path = args.model_path or planner_cfg.get("rl_model_path")
    if not model_path:
        model_path = "BrainSimulationSystem/models/rl/planner_policy.zip"

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    rl_agent_config = RLAgentConfig(
        algorithm="ppo",
        policy=args.policy,
        model_path=model_path,
        device=args.device,
        verbose=int(args.verbose),
    )

    env_factory = build_env_factory(planner_cfg, args.max_options, args.seed)
    agent = DecisionRLAgent(config=rl_agent_config, env_builder=env_factory)

    result = agent.update(total_timesteps=args.timesteps, save_path=model_path)
    if result.get("status") != "updated":
        print(f"Training did not complete successfully: {result}")
        return 1

    agent.save(model_path)
    print(
        f"Saved HybridPlanner RL policy to {model_path} using PPO for {args.timesteps} timesteps "
        f"with profile '{args.profile}'."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
