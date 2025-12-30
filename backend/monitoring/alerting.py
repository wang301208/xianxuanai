"""Simple alerting rules for evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class AlertRule:
    metric: str
    threshold: float
    op: str = "lt"  # "lt" means trigger when metric < threshold, "gt" when >
    message: str | None = None


def evaluate_alerts(metrics: Dict[str, float], rules: List[AlertRule]) -> List[Dict[str, object]]:
    """Return list of alerts for metrics breaching *rules*."""
    alerts: List[Dict[str, object]] = []
    for rule in rules:
        value = metrics.get(rule.metric)
        if value is None:
            continue
        triggered = False
        if rule.op == "lt" and value < rule.threshold:
            triggered = True
        elif rule.op == "gt" and value > rule.threshold:
            triggered = True
        if triggered:
            alerts.append(
                {
                    "metric": rule.metric,
                    "value": value,
                    "message": rule.message or f"{rule.metric} {value} breached {rule.threshold}",
                }
            )
    return alerts
