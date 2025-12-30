"""Validation helpers for self-improvement plans."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ValidationResult:
    approved: bool
    report_path: Path
    details: dict[str, Any]


class PlanValidator:
    def __init__(
        self,
        reports_dir: Path,
        min_success_improvement: float,
        protected_commands: list[str] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._reports_dir = reports_dir
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        self._min_improvement = min_success_improvement
        self._protected_commands = set(protected_commands or [])
        self._logger = logger or logging.getLogger(__name__)

    def validate(
        self,
        baseline_success: float | None,
        current_success: float,
        plan: dict[str, Any] | None,
        metrics: dict[str, Any],
    ) -> ValidationResult:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = self._reports_dir / f"validation_{timestamp}.json"

        baseline = baseline_success if baseline_success is not None else current_success
        delta = current_success - baseline
        approved = delta >= self._min_improvement

        disabled = (plan or {}).get("disabled_commands") or []
        violations = sorted(self._protected_commands.intersection(disabled)) if self._protected_commands else []
        if violations:
            approved = False

        report = {
            "timestamp": timestamp,
            "baseline_success": baseline_success,
            "current_success": current_success,
            "delta": delta,
            "threshold": self._min_improvement,
            "approved": approved,
            "plan_summary": {
                "violations": {"protected_commands": violations} if violations else None,
                "disabled": (plan or {}).get("disabled_commands"),
                "preferred": (plan or {}).get("preferred_commands"),
                "hints": (plan or {}).get("prompt_hints"),
            },
            "metrics": {
                "total_records": metrics.get("total"),
                "command_stats": {
                    name: {
                        "successes": stat.successes,
                        "total": stat.total,
                        "success_rate": stat.success_rate,
                    }
                    for name, stat in metrics.get("command_stats", {}).items()
                }
                if metrics.get("command_stats")
                else None,
            },
        }

        try:
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            self._logger.exception("Failed to write validation report")

        return ValidationResult(approved=approved, report_path=report_path, details=report)
