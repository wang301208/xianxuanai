"""Utilities for generating learning curves and tracking failure cases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import os

import matplotlib.pyplot as plt
import pandas as pd

from monitoring.storage import TimeSeriesStorage


@dataclass
class InterpretabilityAnalyzer:
    """Collects metrics, failure cases and explanations for analysis."""

    failure_cases: List[Dict[str, object]] = field(default_factory=list)
    explanations: List[str] = field(default_factory=list)
    storage: TimeSeriesStorage | None = None

    def generate_learning_curve(self, metrics: List[float], path: str) -> str:
        """Save a learning curve plot for the provided metric values."""

        plt.figure()
        plt.plot(range(len(metrics)), metrics)
        plt.xlabel("iteration")
        plt.ylabel("metric")
        plt.title("Learning Curve")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return path

    def log_failure_case(self, input_data: object, output: object, expected: object) -> None:
        """Record a failure case for later inspection."""

        self.failure_cases.append({
            "input": input_data,
            "output": output,
            "expected": expected,
        })

    def log_explanation(self, text: str) -> None:
        """Record an explanation and persist it via storage if available."""
        self.explanations.append(text)
        if self.storage is not None:
            self.storage.store("analysis.explanations", {"text": text})

    def export_failure_cases(self, path: str) -> str:
        """Write failure cases to a CSV file."""

        df = pd.DataFrame(self.failure_cases)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        return path
