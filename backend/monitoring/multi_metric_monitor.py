"""Collects training, inference and resource metrics and generates visualizations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import os
import time

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    plt = None  # type: ignore

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore


@dataclass
class MultiMetricMonitor:
    """Utility for logging multiple metrics and plotting them.

    The monitor keeps track of training, inference and system resource
    metrics. Training and inference metrics are logged as scalar values per
    step. Resource metrics are logged with timestamps and include CPU and
    memory usage percentages. Arbitrary metric snapshots can also be stored
    for downstream visualisation or export.
    """

    training: List[Tuple[int, float]] = field(default_factory=list)
    inference: List[Tuple[int, float]] = field(default_factory=list)
    resource: List[Tuple[float, Dict[str, float]]] = field(default_factory=list)
    snapshots: List[Tuple[float, Dict[str, float]]] = field(default_factory=list)

    def log_training(self, value: float, step: int | None = None) -> None:
        """Log a training metric value.

        Args:
            value: Metric value to log.
            step: Optional explicit step index. If not provided the next
                integer after the last logged step is used.
        """

        if step is None:
            step = len(self.training)
        self.training.append((step, value))
        self.snapshots.append((time.time(), {"training": float(value)}))

    def log_inference(self, value: float, step: int | None = None) -> None:
        """Log an inference metric value."""

        if step is None:
            step = len(self.inference)
        self.inference.append((step, value))
        self.snapshots.append((time.time(), {"inference": float(value)}))

    def log_resource(self) -> None:
        """Log current system resource usage."""

        usage = {
            "cpu": psutil.cpu_percent(interval=None) if psutil else 0.0,
            "memory": psutil.virtual_memory().percent if psutil else 0.0,
        }
        self.resource.append((time.time(), usage))

    def log_snapshot(self, metrics: Dict[str, float]) -> None:
        """Log a timestamped snapshot of arbitrary metrics."""

        if not metrics:
            return
        try:
            stamped = {k: float(v) for k, v in metrics.items()}
        except Exception:
            stamped = {}
            for k, v in metrics.items():
                try:
                    stamped[k] = float(v)
                except Exception:
                    continue
        if stamped:
            self.snapshots.append((time.time(), stamped))

    def plot(self, out_dir: str) -> List[str]:
        """Generate line plots for collected metrics.

        Args:
            out_dir: Directory to write figures to.

        Returns:
            A list of file paths to the generated figures.
        """

        if plt is None:
            raise RuntimeError("matplotlib is required to plot metrics")

        os.makedirs(out_dir, exist_ok=True)
        created: List[str] = []

        if self.training:
            steps, values = zip(*self.training)
            path = os.path.join(out_dir, "training.png")
            _plot_simple(steps, values, path, "Training Metric")
            created.append(path)

        if self.inference:
            steps, values = zip(*self.inference)
            path = os.path.join(out_dir, "inference.png")
            _plot_simple(steps, values, path, "Inference Metric")
            created.append(path)

        if self.resource:
            timestamps, usage = zip(*self.resource)
            cpu = [u["cpu"] for u in usage]
            mem = [u["memory"] for u in usage]
            path = os.path.join(out_dir, "resource.png")
            plt.figure()
            plt.plot(timestamps, cpu, label="cpu")
            plt.plot(timestamps, mem, label="memory")
            plt.xlabel("time")
            plt.ylabel("percentage")
            plt.title("Resource Usage")
            plt.legend()
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            created.append(path)

        return created


def _plot_simple(x: List[Any], y: List[float], path: str, title: str) -> None:
    """Helper to create a simple line plot."""

    if plt is None:  # pragma: no cover - defensive guard
        raise RuntimeError("matplotlib is required to plot metrics")

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("step")
    plt.ylabel("value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
