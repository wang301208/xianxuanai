from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import json


@dataclass
class RLMetrics:
    reward_gain: float
    guardrail_breaches: int
    evaluation_coverage: float


class RLMetricsTracker:
    """Track Phase 2 reinforcement-learning KPIs."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, metrics: RLMetrics) -> None:
        payload = {
            "reward_gain": metrics.reward_gain,
            "guardrail_breaches": metrics.guardrail_breaches,
            "evaluation_coverage": metrics.evaluation_coverage,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def load(self) -> List[RLMetrics]:
        if not self.log_path.exists():
            return []
        entries: List[RLMetrics] = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            data = json.loads(line)
            entries.append(
                RLMetrics(
                    reward_gain=data.get("reward_gain", 0.0),
                    guardrail_breaches=data.get("guardrail_breaches", 0),
                    evaluation_coverage=data.get("evaluation_coverage", 0.0),
                )
            )
        return entries
