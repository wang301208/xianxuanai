"""Multi-objective fitness helpers.

The evolution stack often needs to collapse multiple signals (success rate,
latency, throughput, energy) into a scalar score for GA/NAS selection.
This module centralises that logic so both the EvolutionEngine and agent-level
self-improvement can share consistent weighting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


@dataclass(frozen=True)
class MultiObjectiveConfig:
    """Weights/targets used to combine metrics."""

    latency_weight: float = 0.1
    energy_weight: float = 0.05
    throughput_weight: float = 0.01
    success_weight: float = 0.0


def summarize_metric_events(metrics: Sequence[Any]) -> Dict[str, float]:
    if not metrics:
        return {"avg_latency": 0.0, "avg_energy": 0.0, "avg_throughput": 0.0, "success_rate": 0.0}

    latencies = [_safe_float(getattr(m, "latency", 0.0)) for m in metrics]
    energies = [_safe_float(getattr(m, "energy", 0.0)) for m in metrics]
    throughputs = [_safe_float(getattr(m, "throughput", 0.0)) for m in metrics]
    statuses = [getattr(m, "status", None) for m in metrics]
    success_count = 0
    total = 0
    for status in statuses:
        if status is None:
            continue
        total += 1
        if str(status).strip().lower() in {"success", "ok", "passed", "complete", "completed"}:
            success_count += 1
    success_rate = success_count / max(total, 1) if total else 0.0

    return {
        "avg_latency": sum(latencies) / max(len(latencies), 1),
        "avg_energy": sum(energies) / max(len(energies), 1),
        "avg_throughput": sum(throughputs) / max(len(throughputs), 1),
        "success_rate": float(success_rate),
    }


def adjust_performance(
    performance: float,
    metrics: Sequence[Any],
    *,
    config: MultiObjectiveConfig | None = None,
) -> Tuple[float, Dict[str, float]]:
    cfg = config or MultiObjectiveConfig()
    summary = summarize_metric_events(metrics)
    avg_latency = summary["avg_latency"]
    avg_energy = summary["avg_energy"]
    avg_throughput = summary["avg_throughput"]
    success_rate = summary["success_rate"]

    penalty = cfg.latency_weight * avg_latency + cfg.energy_weight * avg_energy
    bonus = cfg.throughput_weight * avg_throughput + cfg.success_weight * success_rate
    adjusted = float(performance) - float(penalty) + float(bonus)
    details = {
        "multiobjective_penalty": float(penalty),
        "multiobjective_bonus": float(bonus),
        "avg_latency": float(avg_latency),
        "avg_energy": float(avg_energy),
        "avg_throughput": float(avg_throughput),
        "success_rate": float(success_rate),
    }
    return adjusted, details


__all__ = ["MultiObjectiveConfig", "summarize_metric_events", "adjust_performance"]

