"""Meta-learning trainer for rapid task adaptation across abilities."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.ml.meta_learning.maml import TaskData  # noqa: E402
from backend.ml.meta_learning.trainer import MetaLearningTrainer  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_sequences(df: pd.DataFrame) -> pd.Series:
    def format_part(label: str, value: object) -> str:
        if pd.isna(value) or value is None:
            return ""
        text = str(value).strip()
        return f"[{label}] {text}" if text else ""

    sequences = []
    for row in df.itertuples(index=False):
        parts = [
            format_part("STATE", getattr(row, "state", "")),
            format_part("ABILITY", getattr(row, "ability", "")),
            format_part("INPUT", getattr(row, "input", "")),
            format_part("OUTPUT", getattr(row, "output", "")),
        ]
        reward = getattr(row, "reward", None)
        if reward == reward:
            parts.append(f"[REWARD] {reward}")
        sequences.append(" ".join(part for part in parts if part))
    return pd.Series(sequences, dtype=str)


def discretize_reward(reward: pd.Series, strategy: str = "sign") -> pd.Series:
    if strategy == "quantile":
        quantiles = reward.quantile([0.33, 0.66]).tolist()
        bins = [-np.inf, quantiles[0], quantiles[1], np.inf]
        labels = [0, 1, 2]
    else:
        bins = [-np.inf, -1e-9, 1e-9, np.inf]
        labels = [0, 1, 2]
    return pd.cut(reward, bins=bins, labels=labels).astype(int)


def build_tasks(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    *,
    algorithm: str,
    min_samples: int,
    support_fraction: float,
    seed: int,
    classification_strategy: str,
) -> Tuple[List[TaskData], Dict[str, int]]:
    tasks: List[TaskData] = []
    summary: Dict[str, int] = {}
    for ability, group in df.groupby("ability"):
        group = group.dropna(subset=["sequence", "reward"])
        if len(group) < min_samples:
            continue

        texts = group["sequence"].tolist()
        features = vectorizer.transform(texts).toarray().astype(float)
        if algorithm == "protonet":
            labels = discretize_reward(group["reward"], strategy=classification_strategy).values.astype(float)
            if len(np.unique(labels)) < 2:
                continue
        else:
            labels = group["reward"].astype(float).values

        test_size = max(0.2, 1 - support_fraction)
        try:
            support_x, query_x, support_y, query_y = train_test_split(
                features,
                labels,
                test_size=test_size,
                random_state=seed,
            )
        except ValueError:
            continue

        if len(support_x) == 0 or len(query_x) == 0:
            continue
        tasks.append(TaskData(support_x, support_y, query_x, query_y))
        summary[ability] = len(group)
    return tasks, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train meta-learning model across abilities")
    parser.add_argument(
        "data",
        type=Path,
        help="CSV containing columns 'sequence', 'ability', and 'reward'",
    )
    parser.add_argument("--version", default="meta_v1", help="Artifact version label")
    parser.add_argument("--algorithm", choices=["maml", "reptile", "protonet"], default="maml")
    parser.add_argument("--min-samples", type=int, default=12, help="Minimum samples per ability task")
    parser.add_argument("--support-fraction", type=float, default=0.6, help="Fraction of each task in support set")
    parser.add_argument("--max-features", type=int, default=4096, help="TF-IDF feature limit")
    parser.add_argument("--epochs", type=int, default=30, help="Meta-training epochs")
    parser.add_argument("--inner-lr", type=float, default=0.01, help="Inner-loop learning rate")
    parser.add_argument("--meta-lr", type=float, default=0.001, help="Meta learning rate")
    parser.add_argument("--adapt-steps", type=int, default=1, help="Inner-loop adaptation steps")
    parser.add_argument("--embedding-dim", type=int, default=32, help="Embedding size for protonet")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--classification-reward", choices=["sign", "quantile"], default="sign")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    df = pd.read_csv(args.data)
    if df.empty:
        raise ValueError("Meta-learning dataset is empty")
    if "sequence" not in df.columns or "ability" not in df.columns:
        raise ValueError("Dataset must include 'sequence' and 'ability' columns")
    if "reward" not in df.columns:
        raise ValueError("Dataset must include 'reward' column for targets")

    df["sequence"] = df["sequence"].fillna("")
    df = df[df["sequence"].str.len() > 0]
    if df.empty:
        raise ValueError("All sequences are empty after preprocessing")

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        min_df=2,
    )
    vectorizer.fit(df["sequence"])

    tasks, summary = build_tasks(
        df,
        vectorizer,
        algorithm=args.algorithm,
        min_samples=args.min_samples,
        support_fraction=args.support_fraction,
        seed=args.seed,
        classification_strategy=args.classification_reward,
    )

    if not tasks:
        raise ValueError("No ability groups produced valid meta-learning tasks")

    input_dim = len(vectorizer.get_feature_names_out())
    trainer = MetaLearningTrainer(
        algorithm=args.algorithm,
        input_dim=input_dim,
        inner_lr=args.inner_lr,
        meta_lr=args.meta_lr,
        adapt_steps=args.adapt_steps,
        embedding_dim=args.embedding_dim,
    )

    history = trainer.train(tasks, epochs=args.epochs)
    final_metric = float(history[-1])
    best_metric = float(min(history)) if trainer.metric == "loss" else float(max(history))
    metric_label = "Meta Loss" if trainer.metric == "loss" else "Meta Accuracy"

    artifacts_dir = Path("artifacts") / args.version
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    weights_path = artifacts_dir / "meta_model.npy"
    if hasattr(trainer.model, "weights"):
        np.save(weights_path, getattr(trainer.model, "weights"))
    elif hasattr(trainer.model, "embedding"):
        np.save(weights_path, getattr(trainer.model, "embedding"))

    joblib.dump(vectorizer, artifacts_dir / "tfidf_vectorizer.joblib")

    with open(artifacts_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({"metric": history, "label": metric_label}, f, indent=2)

    metrics: Dict[str, float] = {
        metric_label: final_metric,
        f"{metric_label} (best)": best_metric,
        "Meta Tasks": float(len(tasks)),
        "Meta Abilities": float(len(summary)),
    }
    if trainer.metric == "accuracy":
        metrics.setdefault("Meta Loss", float(max(0.0, 1.0 - final_metric)))

    with open(artifacts_dir / "metrics.txt", "w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    coverage = {
        "algorithm": args.algorithm,
        "abilities": summary,
        "min_samples": args.min_samples,
        "support_fraction": args.support_fraction,
        "epochs": args.epochs,
        "metric_label": metric_label,
    }
    with open(artifacts_dir / "task_summary.json", "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2)

    print(
        f"Meta-training complete using {args.algorithm}. "
        f"{metric_label}: {final_metric:.6f} over {len(tasks)} tasks."
    )


if __name__ == "__main__":
    main()
