from __future__ import annotations

"""Metrics for evaluating the impact of the global workspace on task performance."""

from dataclasses import dataclass
from typing import List


@dataclass
class WorkspaceRun:
    """Metrics for a single task run."""

    task_id: str
    baseline_score: float
    workspace_score: float

    @property
    def improvement(self) -> float:
        """Return improvement from using the global workspace."""
        return self.workspace_score - self.baseline_score


class WorkspaceImpactTracker:
    """Accumulates metrics to assess global workspace contribution."""

    def __init__(self) -> None:
        self.runs: List[WorkspaceRun] = []

    def record(self, task_id: str, baseline_score: float, workspace_score: float) -> None:
        """Record metrics for *task_id* comparing baseline and workspace scores."""
        self.runs.append(
            WorkspaceRun(
                task_id=task_id,
                baseline_score=float(baseline_score),
                workspace_score=float(workspace_score),
            )
        )

    def average_improvement(self) -> float:
        """Return average improvement across all recorded runs."""
        if not self.runs:
            return 0.0
        return sum(run.improvement for run in self.runs) / len(self.runs)
