"""Meta-learning training loop for reinforcement learning policies."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.rl import (  # noqa: E402
    ActorCriticAgent,
    MetaLearner,
    MetaLearningConfig,
    SkillLibrary,
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


def _default_config_path() -> Path:
    return REPO_ROOT / "config" / "rl_tasks.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Meta-learn reinforcement learning policies across task suites"
    )
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--meta-batch", type=int, default=4)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--inner-lr", type=float, default=1e-3)
    parser.add_argument("--meta-lr", type=float, default=1e-2)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--skill-threshold", type=float, default=0.7)
    parser.add_argument("--max-skills", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--version", default="meta_rl_v1")
    parser.add_argument(
        "--log-interval", type=int, default=5, help="Iterations between progress logs"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = _set_seed(args.seed)

    suite = load_task_suite(str(args.config))
    if not suite.tasks:
        raise ValueError("Task suite is empty; please define tasks in the YAML config")

    action_space = suite.action_space
    probe_task = suite.sample(rng=rng)
    from backend.rl.simulated_env import SimulatedTaskEnv  # local import to avoid cycles

    env_probe = SimulatedTaskEnv(probe_task, rng=rng)
    obs = env_probe.reset()
    obs_vector, _ = vectorize_observation(env_probe, obs, action_space)
    agent = ActorCriticAgent(len(obs_vector), len(action_space))

    config = MetaLearningConfig(
        meta_iterations=args.iterations,
        meta_batch_size=args.meta_batch,
        inner_steps=args.inner_steps,
        inner_learning_rate=args.inner_lr,
        meta_learning_rate=args.meta_lr,
        eval_episodes=args.eval_episodes,
        skill_success_threshold=args.skill_threshold,
        max_skill_library_size=args.max_skills,
        seed=args.seed,
    )

    skill_library = SkillLibrary(max_size=args.max_skills)
    meta_learner = MetaLearner(
        agent,
        suite,
        config,
        action_space=action_space,
        skill_library=skill_library,
        rng=rng,
    )

    stats_history = meta_learner.meta_train(log_interval=args.log_interval)
    eval_return, eval_success = meta_learner.evaluate(args.eval_episodes)

    artifacts_dir = Path("artifacts") / args.version
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": agent.state_dict(), "action_space": action_space}, artifacts_dir / "meta_policy.pt")

    history_path = artifacts_dir / "meta_training_stats.json"
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump([stat.to_dict() for stat in stats_history], handle, indent=2)

    meta_learner.skill_library.save(artifacts_dir / "skill_library.json")

    summary = TrainingStats(
        episodes=args.iterations,
        mean_return=float(mean([s.avg_inner_return for s in stats_history]))
        if stats_history
        else 0.0,
        mean_length=0.0,
        success_rate=float(mean([s.avg_success_rate for s in stats_history]))
        if stats_history
        else 0.0,
        eval_return=eval_return,
        eval_success_rate=eval_success,
    )

    with open(artifacts_dir / "metrics.txt", "w", encoding="utf-8") as handle:
        handle.write(f"Meta Avg Return: {summary.mean_return}\n")
        handle.write(f"Meta Success Rate: {summary.success_rate}\n")
        handle.write(f"Eval Return: {summary.eval_return}\n")
        handle.write(f"Eval Success Rate: {summary.eval_success_rate}\n")


if __name__ == "__main__":
    main()

