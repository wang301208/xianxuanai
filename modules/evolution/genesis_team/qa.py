"""Quality assurance agent."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional

from .. import Agent


@dataclass(frozen=True)
class QACheckResult:
    name: str
    passed: bool
    details: str = ""


class QA(Agent):
    """Runs basic quality assurance checks."""

    def perform(self) -> str:
        report = self.run_checks()
        passed = sum(1 for r in report if r.passed)
        failed = [r for r in report if not r.passed]
        if not failed:
            return f"QA checks completed: {passed}/{len(report)} passed."
        failures = "; ".join(f"{r.name}: {r.details}" for r in failed)
        return f"QA checks completed: {passed}/{len(report)} passed; failures: {failures}"

    def run_checks(self, extra_checks: Optional[List[Callable[[], QACheckResult]]] = None) -> List[QACheckResult]:
        checks: List[Callable[[], QACheckResult]] = [
            lambda: _import_check("numpy"),
            lambda: _import_check("pydantic"),
            lambda: _import_check("fastapi"),
            lambda: _import_check("uvicorn"),
            lambda: _import_check("requests"),
        ]
        if extra_checks:
            checks.extend(extra_checks)
        return [check() for check in checks]


def _import_check(module_name: str) -> QACheckResult:
    try:
        module = import_module(module_name)
        version = getattr(module, "__version__", "") or ""
        details = f"version={version}" if version else "imported"
        return QACheckResult(name=f"import:{module_name}", passed=True, details=details)
    except Exception as exc:
        return QACheckResult(name=f"import:{module_name}", passed=False, details=str(exc))


__all__ = ["QA", "QACheckResult"]
