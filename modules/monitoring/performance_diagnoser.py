"""High-level performance diagnostics for multi-metric telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .collector import MetricEvent


@dataclass
class DiagnosticIssue:
    """Description of a detected performance issue."""

    kind: str
    metric: str
    value: float
    threshold: float
    module: Optional[str] = None


class PerformanceDiagnoser:
    """Aggregate metrics and surface bottlenecks or reliability issues."""

    def __init__(
        self,
        *,
        max_latency_s: float = 1.0,
        min_success_rate: float = 0.7,
        min_throughput: float = 1.0,
        max_energy: float = 10.0,
        anomaly_model: Optional[Any] = None,
        anomaly_threshold: float = 3.0,
    ) -> None:
        self.max_latency_s = float(max_latency_s)
        self.min_success_rate = float(min_success_rate)
        self.min_throughput = float(min_throughput)
        self.max_energy = float(max_energy)
        self.anomaly_model = anomaly_model
        self.anomaly_threshold = float(anomaly_threshold)

    # ------------------------------------------------------------------ #
    def diagnose(
        self,
        events: Optional[Iterable[MetricEvent]] = None,
        aggregate: Optional[Dict[str, float]] = None,
    ) -> Dict[str, object]:
        """Return aggregated stats and detected issues."""

        events = list(events or [])
        per_module: Dict[str, Dict[str, float]] = {}
        issues: List[DiagnosticIssue] = []

        if events:
            per_module = self._aggregate_events(events)
            issues.extend(self._detect_module_issues(per_module))
            issues.extend(self._detect_anomalies(events))

        summary = self._compute_summary(per_module, aggregate or {})
        issues.extend(self._detect_global_issues(summary))

        return {
            "summary": summary,
            "modules": per_module,
            "issues": issues,
        }

    # ------------------------------------------------------------------ #
    def _aggregate_events(
        self, events: List[MetricEvent]
    ) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        counts: Dict[str, int] = {}
        for event in events:
            module = event.module
            entry = stats.setdefault(
                module,
                {
                    "latency": 0.0,
                    "throughput": 0.0,
                    "energy": 0.0,
                    "success_count": 0,
                    "total": 0,
                },
            )
            counts[module] = counts.get(module, 0) + 1
            entry["latency"] += float(event.latency)
            entry["throughput"] += float(event.throughput)
            entry["energy"] += float(event.energy)
            status = (event.status or "").lower()
            if status in {"success", "ok", "passed", "complete"}:
                entry["success_count"] += 1
            entry["total"] += 1

        for module, entry in stats.items():
            total = max(entry["total"], 1)
            entry["avg_latency"] = entry["latency"] / total
            entry["avg_throughput"] = entry["throughput"] / total
            entry["avg_energy"] = entry["energy"] / total
            entry["success_rate"] = entry["success_count"] / total
        return stats

    # ------------------------------------------------------------------ #
    def _detect_module_issues(
        self, stats: Dict[str, Dict[str, float]]
    ) -> List[DiagnosticIssue]:
        issues: List[DiagnosticIssue] = []
        for module, entry in stats.items():
            if entry.get("avg_latency", 0.0) > self.max_latency_s:
                issues.append(
                    DiagnosticIssue(
                        kind="high_latency",
                        metric="latency",
                        value=float(entry["avg_latency"]),
                        threshold=self.max_latency_s,
                        module=module,
                    )
                )
            if entry.get("success_rate", 1.0) < self.min_success_rate:
                issues.append(
                    DiagnosticIssue(
                        kind="low_success_rate",
                        metric="success_rate",
                        value=float(entry["success_rate"]),
                        threshold=self.min_success_rate,
                        module=module,
                    )
                )
            if entry.get("avg_throughput", 0.0) < self.min_throughput:
                issues.append(
                    DiagnosticIssue(
                        kind="low_throughput",
                        metric="throughput",
                        value=float(entry["avg_throughput"]),
                        threshold=self.min_throughput,
                        module=module,
                    )
                )
            if entry.get("avg_energy", 0.0) > self.max_energy:
                issues.append(
                    DiagnosticIssue(
                        kind="high_energy",
                        metric="energy",
                        value=float(entry["avg_energy"]),
                        threshold=self.max_energy,
                        module=module,
                    )
                )
        return issues

    # ------------------------------------------------------------------ #
    def _compute_summary(
        self, stats: Dict[str, Dict[str, float]], aggregate: Dict[str, float]
    ) -> Dict[str, float]:
        summary: Dict[str, float] = {}
        if stats:
            latencies = [entry["avg_latency"] for entry in stats.values()]
            throughputs = [entry["avg_throughput"] for entry in stats.values()]
            energies = [entry["avg_energy"] for entry in stats.values()]
            successes: List[Tuple[int, int]] = [
                (entry["success_count"], entry["total"]) for entry in stats.values()
            ]
            total_success = sum(s for s, _ in successes)
            total = sum(t for _, t in successes) or 1
            summary.update(
                {
                    "avg_latency": sum(latencies) / len(latencies),
                    "avg_throughput": sum(throughputs) / len(throughputs),
                    "avg_energy": sum(energies) / len(energies),
                    "success_rate": total_success / total,
                }
            )
        summary.update({k: float(v) for k, v in aggregate.items()})
        return summary

    # ------------------------------------------------------------------ #
    def _detect_global_issues(self, summary: Dict[str, float]) -> List[DiagnosticIssue]:
        issues: List[DiagnosticIssue] = []
        latency = summary.get("avg_latency")
        if latency is not None and latency > self.max_latency_s:
            issues.append(
                DiagnosticIssue(
                    kind="global_high_latency",
                    metric="latency",
                    value=float(latency),
                    threshold=self.max_latency_s,
                    module=None,
                )
            )
        success = summary.get("success_rate")
        if success is not None and success < self.min_success_rate:
            issues.append(
                DiagnosticIssue(
                    kind="global_low_success_rate",
                    metric="success_rate",
                    value=float(success),
                    threshold=self.min_success_rate,
                    module=None,
                )
            )
        throughput = summary.get("avg_throughput")
        if throughput is not None and throughput < self.min_throughput:
            issues.append(
                DiagnosticIssue(
                    kind="global_low_throughput",
                    metric="throughput",
                    value=float(throughput),
                    threshold=self.min_throughput,
                    module=None,
                )
            )
        energy = summary.get("avg_energy")
        if energy is not None and energy > self.max_energy:
            issues.append(
                DiagnosticIssue(
                    kind="global_high_energy",
                    metric="energy",
                    value=float(energy),
                    threshold=self.max_energy,
                    module=None,
                )
            )
        return issues

    # ------------------------------------------------------------------ #
    def _detect_anomalies(self, events: List[MetricEvent]) -> List[DiagnosticIssue]:
        if self.anomaly_model is None:
            return []
        scores = self.anomaly_model.score_events(events)
        issues: List[DiagnosticIssue] = []
        for module, score in scores.items():
            if score >= self.anomaly_threshold:
                issues.append(
                    DiagnosticIssue(
                        kind="anomaly",
                        metric="anomaly_score",
                        value=float(score),
                        threshold=self.anomaly_threshold,
                        module=module,
                    )
                )
        return issues


__all__ = ["PerformanceDiagnoser", "DiagnosticIssue"]
