"""Definitions for offline replay scenarios."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from autogpt.core.learning.experience_store import ExperienceRecord


@dataclass
class ReplayScenario:
    name: str
    description: str
    records: list[ExperienceRecord]

    @classmethod
    def from_json(cls, path: Path) -> "ReplayScenario":
        data = json.loads(path.read_text(encoding="utf-8"))
        records = [ExperienceRecord.from_dict(item) for item in data.get("records", [])]
        return cls(
            name=data.get("name", path.stem),
            description=data.get("description", ""),
            records=records,
        )


class ScenarioLoader:
    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def load(self) -> list[ReplayScenario]:
        if not self._directory.exists():
            return []
        scenarios = []
        for path in self._directory.glob("*.json"):
            try:
                scenarios.append(ReplayScenario.from_json(path))
            except Exception:
                continue
        return scenarios
