"""Automated model retraining pipeline.

This module accumulates new log data, retrains the model, evaluates it
against the current baseline, and deploys the new model if it performs
better.  It is intended to be triggered periodically (for example by a
cron job).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
from events import create_event_bus, publish
from .data_pipeline import DataPipeline

from .compression import compress_model
from .model_registry import ModelRegistry
from . import distributed
from ..memory.long_term import LongTermMemory

DATA_DIR = Path("data")
DATASET = DATA_DIR / "dataset.csv"
NEW_LOGS = DATA_DIR / "new_logs.csv"
ARTIFACTS = Path("artifacts")
CURRENT = ARTIFACTS / "current"
PREVIOUS = ARTIFACTS / "previous"
ML_DIR = Path(__file__).resolve().parent

HISTORY_FILE = Path("evolution/metrics_history.csv")
HISTORY_FIELDS = [
    "timestamp",
    "version",
    "Success Rate",
    "Mean Reward",
    "Perplexity",
    "Test MSE",
    "status",
]

event_bus = create_event_bus()

# Additional metrics considered during deployment comparison. ``direction``
# specifies whether higher values are better ("higher") or lower values are
# better ("lower"). ``threshold`` sets the minimum required improvement.
METRIC_THRESHOLDS: Dict[str, Dict[str, float | str]] = {
    "Success Rate": {"direction": "higher", "threshold": 0.0},
    "Mean Reward": {"direction": "higher", "threshold": 0.0},
    "Eval Return": {"direction": "higher", "threshold": 0.0},
    "Eval Success Rate": {"direction": "higher", "threshold": 0.0},
    "Mean Reconstruction": {"direction": "lower", "threshold": 0.0},
    "Meta Loss": {"direction": "lower", "threshold": 0.0},
    "Meta Accuracy": {"direction": "higher", "threshold": 0.0},
}

DEFAULT_MEMORY_PATH = DATA_DIR / "memory.db"
MEMORY_DB = Path(os.getenv("LONG_TERM_MEMORY_PATH", str(DEFAULT_MEMORY_PATH)))
REFLECTION_CATEGORY = "reflection"
SELF_MONITORING_CATEGORY = "self_monitoring"
TRAINING_INTERACTION_CATEGORY = "training_interaction"
MEMORY_DATASET_COLUMNS = ["state", "ability", "input", "output", "reward"]

logger = logging.getLogger(__name__)


def accumulate_logs() -> None:
    """Append new logs to the main dataset and clear the buffer file.

    New logs are expected in ``data/new_logs.csv`` with columns
    ``state, ability, input, output, reward``. They are appended to
    ``data/dataset.csv`` which serves as the aggregated dataset for training.
    The buffer file is removed afterwards.
    """
    if not NEW_LOGS.exists():
        return

    DATA_DIR.mkdir(exist_ok=True)
    columns = ["state", "ability", "input", "output", "reward"]
    if DATASET.exists():
        df = pd.read_csv(DATASET)
    else:
        df = pd.DataFrame(columns=columns)
    new_df = pd.read_csv(NEW_LOGS)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(DATASET, index=False)
    NEW_LOGS.unlink()


def parse_metric(metrics_file: Path, key: str) -> float:
    """Extract a metric value labelled by ``key`` from ``metrics_file``."""
    with open(metrics_file, "r") as f:
        for line in f:
            if line.startswith(f"{key}:"):
                return float(line.split(":", 1)[1].strip())
    raise ValueError(f"{key} not found in metrics file")


def parse_metrics(metrics_file: Path) -> Dict[str, float]:
    """Parse a metrics file into a dictionary of metric names to values."""
    metrics: Dict[str, float] = {}
    if not metrics_file.exists():
        return metrics
    with open(metrics_file, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            try:
                metrics[key.strip()] = float(value.strip())
            except ValueError:
                continue
    return metrics


def _hash_state(*parts: object) -> str:
    digest = hashlib.sha256(
        "||".join(str(part) for part in parts if part is not None).encode("utf-8")
    ).hexdigest()
    return digest[:16]


def _safe_json_load(payload: str) -> Any:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _reflection_samples(memory: LongTermMemory, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    remaining = max(limit, 0)
    for raw in memory.get(category=REFLECTION_CATEGORY, newest_first=True, limit=limit):
        data = _safe_json_load(raw)
        if not isinstance(data, list):
            continue
        for entry in data:
            if remaining == 0:
                return rows
            evaluation = entry.get("evaluation", {}) if isinstance(entry, dict) else {}
            revision = entry.get("revision", "") if isinstance(entry, dict) else ""
            confidence = float(evaluation.get("confidence", 0.0) or 0.0)
            sentiment = evaluation.get("sentiment", "neutral")
            raw_eval = evaluation.get("raw")
            if not isinstance(raw_eval, str):
                raw_eval = json.dumps(evaluation, ensure_ascii=True, sort_keys=True)
            state = f"reflection:{sentiment}:{_hash_state(raw_eval, revision)}"
            rows.append(
                {
                    "state": state,
                    "ability": f"reflection::{sentiment}",
                    "input": raw_eval,
                    "output": revision,
                    "reward": confidence,
                }
            )
            remaining -= 1
            if remaining == 0:
                break
    return rows


def _self_monitoring_samples(memory: LongTermMemory, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    remaining = max(limit, 0)
    for raw in memory.get(category=SELF_MONITORING_CATEGORY, newest_first=True, limit=limit):
        record = _safe_json_load(raw)
        if not isinstance(record, dict):
            continue
        summary = record.get("summary", "")
        revision = record.get("revision", "")
        evaluation = record.get("evaluation", {})
        confidence = float(evaluation.get("confidence", 0.0) or 0.0)
        sentiment = evaluation.get("sentiment", "neutral")
        state = f"self_monitor:{sentiment}:{_hash_state(summary, revision)}"
        rows.append(
            {
                "state": state,
                "ability": "self_monitoring",
                "input": summary,
                "output": revision,
                "reward": confidence,
            }
        )
        remaining -= 1
        if remaining == 0:
            break
    return rows


def _training_interaction_samples(memory: LongTermMemory, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    remaining = max(limit, 0)
    for raw in memory.get(
        category=TRAINING_INTERACTION_CATEGORY, newest_first=True, limit=limit
    ):
        record = _safe_json_load(raw)
        if not isinstance(record, dict):
            continue
        state = str(record.get("task", "unknown"))
        ability = str(record.get("ability", "unknown"))
        strategy = record.get("strategy")
        reward = float(record.get("reward", 0.0) or 0.0)
        analysis = record.get("analysis", {})
        plan = record.get("plan", {})
        reflection = record.get("reflection", {})
        result = record.get("result", {})
        input_payload = json.dumps(
            {
                "plan": plan,
                "analysis": analysis,
                "reflection": reflection,
            },
            ensure_ascii=True,
            sort_keys=True,
            default=str,
        )
        output_payload = json.dumps(result, ensure_ascii=True, sort_keys=True, default=str)
        rows.append(
            {
                "state": state,
                "ability": f"{ability}:{strategy}" if strategy else ability,
                "input": input_payload,
                "output": output_payload,
                "reward": reward,
            }
        )
        remaining -= 1
        if remaining == 0:
            break
    return rows


def harvest_memory_samples(
    memory_path: Path,
    *,
    limit: int = 900,
) -> List[Dict[str, Any]]:
    """Return training samples distilled from LongTermMemory categories."""

    if not memory_path.exists():
        return []
    memory = LongTermMemory(memory_path)
    try:
        per_bucket = max(limit // 3, 1)
        rows: List[Dict[str, Any]] = []
        rows.extend(_reflection_samples(memory, per_bucket))
        rows.extend(_self_monitoring_samples(memory, per_bucket))
        rows.extend(_training_interaction_samples(memory, limit - len(rows)))
        return rows
    finally:
        memory.close()


def augment_dataset_with_memory(
    memory_path: Path = MEMORY_DB,
    dataset: Path = DATASET,
    *,
    limit: int = 900,
) -> int:
    """Append samples distilled from memory sources to the main dataset.

    Returns the number of rows added.
    """

    rows = harvest_memory_samples(memory_path, limit=limit)
    if not rows:
        return 0
    new_df = pd.DataFrame(rows, columns=MEMORY_DATASET_COLUMNS)
    if dataset.exists():
        existing = pd.read_csv(dataset)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.drop_duplicates(
            subset=["state", "ability", "input", "output"], keep="last", inplace=True
        )
    else:
        combined = new_df
    combined.to_csv(dataset, index=False)
    return len(new_df)


def _log_history(version: str, metrics: Dict[str, float], status: str) -> None:
    """Append metrics to the history CSV."""

    HISTORY_FILE.parent.mkdir(exist_ok=True)
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "version": version,
        "status": status,
    }
    for key in ["Success Rate", "Mean Reward", "Perplexity", "Test MSE"]:
        row[key] = metrics.get(key, float("nan"))

    write_header = not HISTORY_FILE.exists()
    with open(HISTORY_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _prepare_llm_dataset(version: str) -> Path:
    """Create a supervised fine-tuning dataset filtered by implicit rewards.

    Successful interactions are identified through non-negative reward values.
    If no explicitly successful samples are available the function falls back
    to the highest-reward interactions to avoid interrupting the schedule.
    Only the ``input`` and ``output`` columns are kept for LoRA fine-tuning.
    """

    if not DATASET.exists():
        raise FileNotFoundError("Aggregated dataset not found")

    df = pd.read_csv(DATASET)
    if "reward" in df.columns:
        rewards = pd.to_numeric(df["reward"], errors="coerce").fillna(0.0)
        df = df.assign(reward=rewards, success=rewards >= 0.0)
        successful = df[df["success"]]
        if successful.empty:
            df = df.sort_values("reward", ascending=False).head(200)
        else:
            df = successful

    df = df.dropna(subset=["input", "output"])
    if df.empty:
        raise ValueError("No valid samples available for LLM fine-tuning")

    processed = df[["input", "output"]].copy()
    processed["input"] = processed["input"].astype(str).str.strip()
    processed["output"] = processed["output"].astype(str).str.strip()

    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = processed_dir / f"llm_dataset_{version}.csv"
    processed.to_csv(dataset_path, index=False)
    return dataset_path


def _prepare_self_supervised_dataset(version: str) -> Path:
    """Assemble an unlabeled corpus for self-supervised representation learning."""

    if not DATASET.exists():
        raise FileNotFoundError("Aggregated dataset not found")

    df = pd.read_csv(DATASET)
    if df.empty:
        raise ValueError("Dataset is empty; cannot build self-supervised corpus")

    def to_text(value: object, label: str) -> str:
        if pd.isna(value) or value is None:
            return ""
        text = str(value).strip()
        return f"[{label}] {text}" if text else ""

    sequences = []
    for row in df.itertuples(index=False):
        parts = [
            to_text(getattr(row, "state", ""), "STATE"),
            to_text(getattr(row, "ability", ""), "ABILITY"),
            to_text(getattr(row, "input", ""), "INPUT"),
            to_text(getattr(row, "output", ""), "OUTPUT"),
        ]
        reward = getattr(row, "reward", None)
        if reward == reward:  # check for NaN
            parts.append(f"[REWARD] {reward}")
        sequences.append(" ".join(part for part in parts if part))

    processed = pd.DataFrame(
        {
            "sequence": sequences,
            "state": df.get("state", pd.Series(dtype=str)).fillna("").astype(str),
            "ability": df.get("ability", pd.Series(dtype=str)).fillna("").astype(str),
            "reward": df.get("reward", pd.Series(dtype=float)),
        }
    )
    processed = processed[processed["sequence"].str.len() > 0]
    if processed.empty:
        raise ValueError("No textual content available for self-supervised corpus")

    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = processed_dir / f"selfsup_dataset_{version}.csv"
    processed.to_csv(dataset_path, index=False)
    return dataset_path


def _prepare_meta_dataset(version: str) -> Path:
    """Materialize a dataset tailored for meta-learning tasks."""

    base_path = _prepare_self_supervised_dataset(version)
    meta_path = DATA_DIR / "processed" / f"meta_dataset_{version}.csv"
    shutil.copy(base_path, meta_path)
    return meta_path


def train_and_evaluate(version: str, model: str) -> tuple[Path, str]:
    """Train a model using the aggregated dataset.

    Returns the metrics file path and the metric label to compare.
    """
    if model == "llm":
        dataset_path = _prepare_llm_dataset(version)
        subprocess.run(
            [
                sys.executable,
                str(ML_DIR / "fine_tune_llm.py"),
                str(dataset_path),
                "--version",
                version,
            ],
            check=True,
        )
        metric_name = "Perplexity"
    elif model == "rl":
        subprocess.run(
            [
                sys.executable,
                str(ML_DIR / "train_rl_policy.py"),
                "--version",
                version,
            ],
            check=True,
        )
        metric_name = "Eval Return"
    elif model == "self_supervised":
        dataset_path = _prepare_self_supervised_dataset(version)
        subprocess.run(
            [
                sys.executable,
                str(ML_DIR / "train_self_supervised.py"),
                str(dataset_path),
                "--version",
                version,
            ],
            check=True,
        )
        metric_name = "Mean Reconstruction"
    elif model == "meta":
        dataset_path = _prepare_meta_dataset(version)
        subprocess.run(
            [
                sys.executable,
                str(ML_DIR / "train_meta_learner.py"),
                str(dataset_path),
                "--version",
                version,
            ],
            check=True,
        )
        metric_name = "Meta Loss"
    else:
        subprocess.run(
            [
                sys.executable,
                str(ML_DIR / "train_models.py"),
                str(DATASET),
                "--model",
                model,
                "--version",
                version,
            ],
            check=True,
        )
        metric_name = "Test MSE"
    return ARTIFACTS / version / "metrics.txt", metric_name


def deploy_with_ab_test(
    version: str,
    new_metrics: Dict[str, float],
    metric_name: str,
    registry: ModelRegistry,
) -> bool:
    """Deploy the trained model and run a simple A/B test against the baseline.

    The current model is backed up, the new model is copied into ``CURRENT`` and
    compared against the previous metrics. If any regression is detected the
    deployment is rolled back and the previous model restored.
    """

    baseline_meta = registry.current()
    baseline_metrics = baseline_meta["metrics"] if baseline_meta else {}

    # Deploy new model by swapping directories so that the A/B test can run.
    if CURRENT.exists():
        if PREVIOUS.exists():
            shutil.rmtree(PREVIOUS)
        shutil.copytree(CURRENT, PREVIOUS)
        shutil.rmtree(CURRENT)
    shutil.copytree(ARTIFACTS / version, CURRENT)

    thresholds = METRIC_THRESHOLDS.copy()
    thresholds.setdefault(metric_name, {"direction": "lower", "threshold": 0.0})

    regressions: list[str] = []
    for metric, cfg in thresholds.items():
        if metric not in new_metrics or metric not in baseline_metrics:
            continue
        new_val = new_metrics[metric]
        base_val = baseline_metrics[metric]
        direction = cfg["direction"]
        threshold = float(cfg["threshold"])
        if direction == "lower":
            if new_val > base_val + threshold:
                regressions.append(f"{metric} {new_val:.4f} > {base_val:.4f}")
        else:
            if new_val < base_val - threshold:
                regressions.append(f"{metric} {new_val:.4f} < {base_val:.4f}")

    if not regressions:
        metric_val = new_metrics.get(metric_name, float("nan"))
        print(
            f"Deployed new model version {version} ({metric_name} {metric_val:.4f})",
        )
        registry.set_current(version)
        _log_history(version, new_metrics, "deployed")
        return True

    logger.warning(
        "Model not deployed due to metric regressions: %s", "; ".join(regressions)
    )
    _log_history(version, new_metrics, "regression")
    publish(
        event_bus,
        "model.regression",
        {"version": version, "regressions": regressions},
    )

    # Roll back to the previous model
    if CURRENT.exists():
        shutil.rmtree(CURRENT)
    if PREVIOUS.exists():
        shutil.copytree(PREVIOUS, CURRENT)
    registry.rollback()
    return False


def main() -> bool:
    parser = argparse.ArgumentParser(description="Retrain models on accumulated data")
    parser.add_argument(
        "--model",
        default="linear",
        help="Model type to train (e.g. 'linear' or 'llm')",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=None,
        help="Optional compression level to apply to the trained model",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training hooks",
    )
    args = parser.parse_args()

    registry = ModelRegistry()

    accumulate_logs()
    added_from_memory = augment_dataset_with_memory()
    if added_from_memory:
        logger.info("harvested %d samples from memory stores", added_from_memory)
    if not DATASET.exists():
        print("No dataset available for training")
        return True

    # Expand the dataset using the data pipeline before training.
    df = pd.read_csv(DATASET)
    pipeline = DataPipeline()
    df = pipeline.process(df)
    df.to_csv(DATASET, index=False)

    version = datetime.utcnow().strftime("v%Y%m%d%H%M%S")
    if args.distributed:
        distributed.setup_training()
    metrics_file, metric_name = train_and_evaluate(version, args.model)
    if args.distributed:
        distributed.teardown_training()

    new_metrics = parse_metrics(metrics_file)
    if args.compression_level is not None:
        compress_model(ARTIFACTS / version, args.compression_level)

    registry.register(version, new_metrics, args.compression_level)
    return deploy_with_ab_test(version, new_metrics, metric_name, registry)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
