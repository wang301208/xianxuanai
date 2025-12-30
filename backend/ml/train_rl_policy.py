"""Train reinforcement learning policies inside simulated task environments."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean
from typing import List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.rl import (  # noqa: E402
    ActorCriticAgent,
    SimulatedTaskEnv,
    TaskSuite,
    TrainingStats,
    load_task_suite,
    vectorize_observation,
)


def _set_seed(seed: int) -> random.Random:
    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return rng


def _train_episode(
    agent: ActorCriticAgent,
    env: SimulatedTaskEnv,
    action_space: Sequence[str],
) -> Tuple[float, int, bool, Tuple[float, float, float]]:
    observation = env.reset()
    transitions: List[Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]] = []
    total_reward = 0.0
    steps = 0
    success = False

    for _ in range(env.max_steps):
        features, mask = vectorize_observation(env, observation, action_space)
        obs_tensor = torch.from_numpy(features).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        action_idx, log_prob, value, entropy = agent.act(obs_tensor, mask_tensor)
        action = action_space[action_idx]
        next_obs, reward, done, info = env.step(action)
        total_reward += reward
        transitions.append((log_prob, value, reward, entropy))
        observation = next_obs
        steps += 1
        if done:
            success = bool(info.get("success", 0.0))
            break
    loss_terms = agent.update(transitions)
    return total_reward, steps, success, loss_terms


def _evaluate_policy(
    agent: ActorCriticAgent,
    suite: TaskSuite,
    action_space: Sequence[str],
    rng: random.Random,
    episodes: int,
) -> Tuple[float, float]:
    returns: List[float] = []
    successes = 0
    for _ in range(episodes):
        task = suite.sample(rng=rng)
        env = SimulatedTaskEnv(task, rng=rng)
        obs = env.reset()
        episode_reward = 0.0
        for _ in range(env.max_steps):
            features, mask = vectorize_observation(env, obs, action_space)
            obs_tensor = torch.from_numpy(features).unsqueeze(0)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)
            with torch.no_grad():
                action_idx, _, _, _ = agent.act(obs_tensor, mask_tensor, deterministic=True)
            next_obs, reward, done, info = env.step(action_space[action_idx])
            episode_reward += reward
            obs = next_obs
            if done:
                successes += int(info.get("success", 0.0))
                break
        returns.append(episode_reward)
    avg_return = float(mean(returns)) if returns else 0.0
    success_rate = float(successes) / float(max(1, episodes))
    return avg_return, success_rate


def _default_config_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "config" / "rl_tasks.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL policy in simulated environments")
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config_path(),
        help="Path to YAML file describing task suite",
    )
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--version", default="rl_v1", help="Artifact version label")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = _set_seed(args.seed)

    suite = load_task_suite(str(args.config))
    if not suite.tasks:
        raise ValueError("Task suite is empty; please define tasks in the YAML config")

    action_space = suite.action_space
    # Peek at first task to size the network.
    sample_task = suite.sample(rng=rng)
    env_probe = SimulatedTaskEnv(sample_task, rng=rng)
    probe_obs = env_probe.reset()
    obs_vector, mask = vectorize_observation(env_probe, probe_obs, action_space)
    agent = ActorCriticAgent(len(obs_vector), len(action_space))

    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    successes = 0
    loss_history: List[float] = []

    for episode in range(1, args.episodes + 1):
        task = suite.sample(rng=rng)
        env = SimulatedTaskEnv(task, rng=rng)
        reward, length, success, loss_terms = _train_episode(agent, env, action_space)
        episode_returns.append(reward)
        episode_lengths.append(length)
        successes += int(success)
        loss_history.append(loss_terms[0])

        if episode % 25 == 0 or episode == 1:
            avg_ret = mean(episode_returns[-25:]) if episode_returns else 0.0
            print(
                f"[episode={episode}] reward={reward:.3f} "
                f"avg_return_25={avg_ret:.3f} length={length} success={success} loss={loss_terms[0]:.4f}"
            )

    eval_return, eval_success = _evaluate_policy(
        agent, suite, action_space, rng, episodes=args.eval_episodes
    )
    stats = TrainingStats(
        episodes=args.episodes,
        mean_return=float(mean(episode_returns)) if episode_returns else 0.0,
        mean_length=float(mean(episode_lengths)) if episode_lengths else 0.0,
        success_rate=float(successes) / float(max(1, args.episodes)),
        eval_return=eval_return,
        eval_success_rate=eval_success,
    )

    artifacts_dir = Path("artifacts") / args.version
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    policy_path = artifacts_dir / "rl_policy.pt"
    torch.save({"state_dict": agent.state_dict(), "action_space": action_space}, policy_path)

    with open(artifacts_dir / "training_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats.__dict__, f, indent=2)

    with open(artifacts_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Average Return: {stats.mean_return}\n")
        f.write(f"Average Length: {stats.mean_length}\n")
        f.write(f"Success Rate: {stats.success_rate}\n")
        f.write(f"Eval Return: {stats.eval_return}\n")
        f.write(f"Eval Success Rate: {stats.eval_success_rate}\n")


if __name__ == "__main__":
    main()
