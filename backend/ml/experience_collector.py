from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Set

try:
    from .continual_trainer import ContinualTrainer
except Exception:  # pragma: no cover - fallback when torch unavailable
    class ContinualTrainer:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            self.enabled = False

        def add_sample(self, _sample):
            return None

from . import DEFAULT_TRAINING_CONFIG

LOG_FILE = Path("data") / "new_logs.csv"

# Single trainer instance used to schedule periodic fine-tuning
TRAINER = ContinualTrainer(DEFAULT_TRAINING_CONFIG, LOG_FILE)


class ActiveCuriositySelector:
    """Filter samples based on reward and novelty.

    A simple heuristic combines the running reward average with whether the
    underlying state has been seen before. Samples that are either novel or
    yield above-average reward are forwarded to the trainer.
    """

    def __init__(self, reward_threshold: float = 0.0, novelty_weight: float = 0.5) -> None:
        self.reward_threshold = reward_threshold
        self.novelty_weight = novelty_weight
        self.seen_states: Set[str] = set()
        self.avg_reward = 0.0
        self.count = 0

    def consider(self, sample: Dict[str, Any]) -> bool:
        self.count += 1
        r = sample["reward"]
        # update moving average of reward
        self.avg_reward += (r - self.avg_reward) / self.count
        novelty = 0.0 if sample["state"] in self.seen_states else 1.0
        curiosity = self.novelty_weight * novelty + (1 - self.novelty_weight) * max(
            0.0, r - self.avg_reward
        )
        if curiosity > self.reward_threshold:
            self.seen_states.add(sample["state"])
            return True
        return False


SELECTOR = ActiveCuriositySelector()


def log_interaction(task: Any, ability: str, result: Any, reward: float) -> None:
    """Record an interaction to ``data/new_logs.csv``.

    Parameters
    ----------
    task:
        The task or state associated with this interaction.
    ability:
        Name of the ability that was executed.
    result:
        Result object from the ability execution. ``input`` and ``output``
        attributes (or keys if ``result`` is a mapping) are extracted if
        available.
    reward:
        Numeric reward representing the quality of the result.
    """
    state = getattr(task, "id", getattr(task, "name", str(task)))

    if isinstance(result, dict):
        input_data = result.get("input") or result.get("prompt") or ""
        output_data = result.get("output") or result.get("response") or str(result)
    else:
        input_data = getattr(result, "input", "")
        output_data = getattr(result, "output", str(result))

    sample = {
        "state": state,
        "ability": ability,
        "input": input_data,
        "output": output_data,
        "reward": reward,
    }

    LOG_FILE.parent.mkdir(exist_ok=True)
    file_exists = LOG_FILE.exists()
    with LOG_FILE.open("a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["state", "ability", "input", "output", "reward"]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(sample)

    if SELECTOR.consider(sample):
        TRAINER.add_sample(sample)
