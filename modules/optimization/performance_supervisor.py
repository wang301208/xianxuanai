"""Coordinate diagnostics with evolution and self-improvement."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


class PerformanceSupervisor:
    """Bridge diagnostics to evolution/self-improvement actions."""

    def __init__(
        self,
        *,
        evolution_engine: Any = None,
        self_improvement: Any = None,
    ) -> None:
        self.evolution_engine = evolution_engine
        self.self_improvement = self_improvement

    def handle_issues(self, issues: Iterable[Any]) -> None:
        issues = list(issues)
        if not issues:
            return
        if self.self_improvement is not None and hasattr(self.self_improvement, "record_diagnostics"):
            try:
                self.self_improvement.record_diagnostics(issues)
            except Exception:
                pass
        if self.evolution_engine is not None:
            # Trigger a light evolution cycle to search for structural/param fixes.
            try:
                self.evolution_engine.run_evolution_cycle(metrics=[], benchmarks=None, task=None)
            except Exception:
                pass
