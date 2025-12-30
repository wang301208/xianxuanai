"""Learning utilities for adapting the agent from past experiences."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from autogpt.core.configuration.learning import LearningConfiguration
from autogpt.core.learning.experience_store import ExperienceRecord


class ExperienceLearner:
    """Learn from stored experiences to update model parameters."""

    def __init__(
        self,
        memory: Iterable,
        config: LearningConfiguration,
        logger: logging.Logger | None = None,
    ) -> None:
        self._memory = memory
        self._config = config
        self._logger = logger or logging.getLogger(__name__)
        self._model_path = Path(self._config.model_state_path)
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        self._model, self._trained_records = self._load_model()

    def learn_from_experience(self) -> dict[str, float]:
        """Read past interactions from memory and update the model.

        Returns:
            dict[str, float]: Mapping of command names to their learned
                success weights. Returns the current model state even when no
                update is performed so callers can always rely on the result.
        """
        if not self._config.enabled:
            return self._model

        records = list(self._memory) if self._memory is not None else []
        if not records or len(records) <= self._trained_records:
            return self._model

        new_records = records[self._trained_records :]

        self._logger.debug(
            "Learning from %d records (lr=%s, batch_size=%s)",
            len(new_records),
            self._config.learning_rate,
            self._config.batch_size,
        )

        lr = self._config.learning_rate
        batch_size = self._config.batch_size
        for i in range(0, len(new_records), batch_size):
            batch = new_records[i : i + batch_size]
            for episode in batch:
                command_name, status = self._extract_signal(episode)
                if not command_name or status is None:
                    continue

                value = self._model.get(command_name, 0.0)
                target = 1.0 if status == "success" else 0.0
                value += lr * (target - value)
                self._model[command_name] = value

        self._trained_records = len(records)
        self._save_model()

        return self._model

    def _extract_signal(self, episode) -> tuple[str | None, str | None]:
        if isinstance(episode, ExperienceRecord):
            return episode.command_name, episode.result_status

        action = getattr(episode, "action", None)
        result = getattr(episode, "result", None)
        command_name = getattr(action, "name", None)
        status = getattr(result, "status", None)
        return command_name, status

    def _load_model(self) -> tuple[dict[str, float], int]:
        if self._model_path.exists():
            try:
                with self._model_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("weights", {}), data.get("record_count", 0)
            except Exception:
                self._logger.exception("Failed to load experience model state")
        return {}, 0

    def _save_model(self) -> None:
        data = {"weights": self._model, "record_count": self._trained_records}
        try:
            with self._model_path.open("w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            self._logger.exception("Failed to save experience model state")
