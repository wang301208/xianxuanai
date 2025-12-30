#!/usr/bin/env python3
"""Train a simple meta-learner (MAML/Reptile) on synthetic sine-wave tasks."""

from __future__ import annotations

import argparse
import math
import random
from typing import Dict, List, Tuple

import numpy as np

try:  # pragma: no cover - torch optional
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This training script requires PyTorch. Please install torch before running.") from exc

from BrainSimulationSystem.config.default_config import get_config
from BrainSimulationSystem.learning.meta_learning import (
    MetaLearningConfig,
    MetaLearner,
    MetaLearningGAAdapter,
    MetaTask,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="prototype", help="Configuration profile from default_config.")
    parser.add_argument("--iterations", type=int, default=None, help="Meta-training iterations (overrides config).")
    parser.add_argument("--algorithm", choices=["maml", "reptile"], default=None, help="Meta-learning algorithm.")
    parser.add_argument("--inner-steps", type=int, default=None, help="Number of inner-loop updates.")
    parser.add_argument("--meta-batch", type=int, default=None, help="Number of tasks per meta-iteration.")
    parser.add_argument("--support-size", type=int, default=10, help="Number of support samples per task.")
    parser.add_argument("--query-size", type=int, default=10, help="Number of query samples per task.")
    parser.add_argument("--tasks", type=int, default=20, help="Total number of meta-tasks to sample.")
    parser.add_argument("--hidden", type=int, default=40, help="Hidden width for the regression network.")
    parser.add_argument("--save-path", default=None, help="Optional path to save the trained meta-parameters.")
    parser.add_argument("--use-ga", action="store_true", help="Enable GA-based hyperparameter search before training.")
    parser.add_argument("--ga-generations", type=int, default=8, help="Number of GA generations if --use-ga is set.")
    parser.add_argument("--device", default=None, help="Torch device; defaults to CUDA if available.")
    return parser


def sine_task_factory(
    amplitude: float,
    phase: float,
    frequency: float,
    noise: float,
    *,
    support_size: int,
    query_size: int,
    device: torch.device,
) -> MetaTask:
    """Create a sine-wave regression task."""

    def sample(num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.rand(num_samples, 1, device=device) * 10.0 - 5.0
        y = amplitude * torch.sin(frequency * x + phase)
        if noise > 0.0:
            y += noise * torch.randn_like(y)
        return x, y

    return MetaTask(
        task_id=f"sine_{amplitude:.2f}_{phase:.2f}_{frequency:.2f}",
        support_sampler=lambda: sample(support_size),
        query_sampler=lambda: sample(query_size),
        metadata={
            "amplitude": amplitude,
            "phase": phase,
            "frequency": frequency,
            "noise": noise,
        },
    )


def build_tasks(
    num_tasks: int,
    *,
    support_size: int,
    query_size: int,
    device: torch.device,
    seed: int = 7,
) -> List[MetaTask]:
    rng = random.Random(seed)
    tasks: List[MetaTask] = []
    for _ in range(num_tasks):
        amplitude = rng.uniform(0.1, 5.0)
        phase = rng.uniform(0.0, math.pi)
        frequency = rng.uniform(0.5, 1.5)
        noise = rng.uniform(0.0, 0.1)
        tasks.append(
            sine_task_factory(
                amplitude=amplitude,
                phase=phase,
                frequency=frequency,
                noise=noise,
                support_size=support_size,
                query_size=query_size,
                device=device,
            )
        )
    return tasks


def model_factory(hidden: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(1, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )


def regression_loss(model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    x, y = batch
    preds = model(x)
    return torch.mean((preds - y) ** 2)


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    base_config = get_config(args.profile).get("meta_learning", {})
    config_kwargs: Dict[str, float] = {}
    if base_config:
        config_kwargs.update({k: v for k, v in base_config.get("trainer", {}).items()})

    if args.algorithm:
        config_kwargs["algorithm"] = args.algorithm
    if args.inner_steps is not None:
        config_kwargs["inner_steps"] = int(args.inner_steps)
    if args.meta_batch is not None:
        config_kwargs["meta_batch_size"] = int(args.meta_batch)
    if args.device:
        config_kwargs["device"] = args.device

    config = MetaLearningConfig(**config_kwargs)
    if args.iterations is not None:
        config.meta_iterations = int(args.iterations)

    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tasks = build_tasks(
        args.tasks,
        support_size=args.support_size,
        query_size=args.query_size,
        device=device,
        seed=config.seed or 7,
    )

    learner = MetaLearner(
        model_factory=lambda: model_factory(args.hidden),
        loss_fn=regression_loss,
        config=config,
    )

    if args.use_ga:
        search_space = {
            "inner_learning_rate": {"min": 1e-3, "max": 5e-1},
            "meta_learning_rate": {"min": 1e-4, "max": 5e-2},
            "inner_steps": {"min": 1, "max": 10, "type": "int"},
        }

        def evaluator(hparams: MetaLearningConfig) -> float:
            trial_learner = MetaLearner(
                model_factory=lambda: model_factory(args.hidden),
                loss_fn=regression_loss,
                config=hparams,
            )
            history = trial_learner.meta_train(tasks, iterations=max(3, hparams.meta_iterations // 5))
            return -history[-1].query_loss  # maximise negative loss

        ga_adapter = MetaLearningGAAdapter(config, search_space, evaluator)
        best_config, score = ga_adapter.optimise(generations=args.ga_generations)
        print(f"[GA] Best config: {best_config} score={score:.4f}")
        learner.config = best_config
        learner.reset_model()

    history = learner.meta_train(tasks, iterations=config.meta_iterations)
    print("Meta-training complete.")
    for stats in history[-min(5, len(history)) :]:
        print(
            f"[iter {stats.iteration:03d}] support_loss={stats.support_loss:.6f} "
            f"query_loss={stats.query_loss:.6f} step_norm={stats.step_norm:.4f}"
        )

    # Demonstrate fast adaptation on a new task
    eval_task = sine_task_factory(
        amplitude=2.5,
        phase=0.5,
        frequency=1.0,
        noise=0.05,
        support_size=args.support_size,
        query_size=args.query_size,
        device=device,
    )
    pre_loss = learner.evaluate_task(eval_task, steps=0)
    post_adapt_model = learner.adapt_task(eval_task, steps=learner.config.task_adaptation_steps)
    with torch.no_grad():
        post_loss = regression_loss(post_adapt_model, eval_task.sample_query()).item()

    print(f"Adaptation performance -- pre-adapt loss: {pre_loss:.6f}, post-adapt loss: {post_loss:.6f}")

    if args.save_path:
        torch.save({"state_dict": learner.model.state_dict(), "config": learner.config.__dict__}, args.save_path)
        print(f"Saved meta-learner parameters to {args.save_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
