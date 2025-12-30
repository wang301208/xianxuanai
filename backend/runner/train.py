from __future__ import annotations

import argparse
import yaml

from .logging_config import get_logger
from backend.ml.meta_learning import MetaLearningTrainer, load_task


logger = get_logger(__name__)


def run_meta_learning(
    config_path: str,
    algorithm: str | None,
    shots: int | None,
    ways: int | None,
) -> None:
    """Run meta-learning based on the given YAML configuration."""

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    tasks_cfg = cfg.get("tasks", [])
    if not tasks_cfg:
        raise ValueError("No tasks specified in configuration")

    algorithm = (algorithm or cfg.get("algorithm", "maml")).lower()
    shots = shots if shots is not None else cfg.get("shots")
    ways = ways if ways is not None else cfg.get("ways")

    tasks = [
        load_task(task["dataset"], k_shot=shots, n_way=ways)
        for task in tasks_cfg
    ]
    input_dim = tasks[0].support_x.shape[1]

    trainer = MetaLearningTrainer(
        algorithm=algorithm,
        input_dim=input_dim,
        inner_lr=cfg.get("inner_lr", 0.01),
        meta_lr=cfg.get("meta_lr", 0.001),
        adapt_steps=cfg.get("adapt_steps", 1),
        embedding_dim=cfg.get("embedding_dim", 16),
    )

    history = trainer.train(tasks, epochs=cfg.get("epochs", 1))
    for epoch, metric in enumerate(history, 1):
        logger.info(
            "epoch_metric",
            epoch=epoch,
            metric_name=trainer.metric,
            metric_value=metric,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Training utilities")
    parser.add_argument("--meta", action="store_true", help="Run meta-learning")
    parser.add_argument("--algorithm", type=str, help="Meta-learning algorithm to use")
    parser.add_argument(
        "--shots", type=int, help="Number of examples per class in the support set"
    )
    parser.add_argument(
        "--ways", type=int, help="Number of classes per task"
    )
    parser.add_argument(
        "--config", type=str, default="config/meta_learning.yaml", help="Config path"
    )
    args = parser.parse_args()

    if args.meta:
        run_meta_learning(args.config, args.algorithm, args.shots, args.ways)
    else:
        logger.info(
            "standard_training_not_implemented",
            meta_required=True,
        )


if __name__ == "__main__":
    main()
