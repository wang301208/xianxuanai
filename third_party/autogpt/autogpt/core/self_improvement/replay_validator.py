"""Replay validation logic for self-improvement."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

from autogpt.core.learning.experience_store import ExperienceRecord
from .replay import ReplayScenario


class ReplayValidator:
    def __init__(
        self,
        reports_dir: Path,
        logger: logging.Logger | None = None,
    ) -> None:
        self._reports_dir = reports_dir
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logger or logging.getLogger(__name__)

    def evaluate_scenarios(
        self,
        scenarios: Iterable[ReplayScenario],
        executor,
    ) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for scenario in scenarios:
            try:
                success = executor(scenario)
            except Exception:
                success = False
            results[scenario.name] = success
            self._write_report(scenario, success)
        return results

    def _write_report(self, scenario: ReplayScenario, success: bool) -> None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = self._reports_dir / f"replay_{scenario.name}_{timestamp}.json"
        report = {
            "scenario": scenario.name,
            "description": scenario.description,
            "success": success,
            "records": [record.__dict__ for record in scenario.records],
        }
        try:
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            self._logger.exception("Failed to write replay report")
