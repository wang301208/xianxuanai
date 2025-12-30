from __future__ import annotations

"""Utilities for recording optimization metrics and exporting them."""

from dataclasses import dataclass, asdict
import json
import csv
from pathlib import Path
from typing import Any, List, Optional


@dataclass
class RunMetrics:
    """Container for metrics from a single optimization run."""

    algorithm: str
    problem: str
    seed: int
    best_val: float
    relative_error: float
    iterations: int
    time: float
    iter_limit_reached: bool
    time_limit_reached: bool
    extra: Optional[Any] = None


class MetricsRecorder:
    """Accumulates run metrics and saves them to JSON, CSV, or YAML."""

    def __init__(self) -> None:
        self.records: List[RunMetrics] = []

    def record(
        self,
        algorithm: str,
        problem: str,
        seed: int,
        best_val: float,
        optimum_val: float,
        iterations: int,
        elapsed_time: float,
        max_iters: Optional[int] = None,
        max_time: Optional[float] = None,
        extra: Optional[Any] = None,
    ) -> None:
        """Record metrics for a single run.

        ``max_iters`` and ``max_time`` are used to determine whether the run hit
        the iteration or time budget.
        """
        if optimum_val == 0:
            relative_error = abs(best_val - optimum_val)
        else:
            relative_error = abs(best_val - optimum_val) / abs(optimum_val)

        iter_limit_reached = max_iters is not None and iterations >= max_iters
        time_limit_reached = max_time is not None and elapsed_time >= max_time

        self.records.append(
            RunMetrics(
                algorithm=algorithm,
                problem=problem,
                seed=seed,
                best_val=float(best_val),
                relative_error=float(relative_error),
                iterations=int(iterations),
                time=float(elapsed_time),
                iter_limit_reached=iter_limit_reached,
                time_limit_reached=time_limit_reached,
                extra=extra,
            )
        )

    def save(self, path: str) -> None:
        """Save all recorded metrics to ``path`` as JSON, CSV, or YAML.

        Parent directories are created automatically to mirror standard logging
        behavior when users point the recorder at a nested output location.
        """
        path_obj = Path(path)
        if path_obj.parent:
            path_obj.parent.mkdir(parents=True, exist_ok=True)

        records = [asdict(r) for r in self.records]
        lower = path_obj.name.lower()
        if lower.endswith(".json"):
            with path_obj.open("w") as f:
                json.dump(records, f, indent=2)
        elif lower.endswith(".csv"):
            flattened = [_flatten_dict(r) for r in records]
            fieldnames: List[str] = sorted({k for rec in flattened for k in rec.keys()})
            with path_obj.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for rec in flattened:
                    writer.writerow({fn: rec.get(fn, "") for fn in fieldnames})
        elif lower.endswith(".yaml") or lower.endswith(".yml"):
            try:
                import yaml  # type: ignore
            except Exception as e:  # pragma: no cover - handled in tests
                raise ValueError("PyYAML is required for YAML serialization") from e
            with path_obj.open("w") as f:
                yaml.safe_dump(records, f)
        else:
            raise ValueError("Unsupported file format: expected .json, .csv, or .yaml")

    def to_list(self) -> List[dict]:
        """Return recorded metrics as list of dictionaries."""
        return [asdict(r) for r in self.records]


def _flatten_dict(data: Any, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested ``dict``/``list`` using dot-separated keys."""
    items: dict[str, Any] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            items.update(_flatten_dict(v, new_key, sep=sep))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(_flatten_dict(v, new_key, sep=sep))
    else:
        items[parent_key] = data
    return items
