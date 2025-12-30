"""Founder agent that observes system metrics and suggests improvements."""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None

from monitoring import ActionLogger

from . import Agent
from .genesis_team import GenesisTeamManager
from .policy_model import FeatureExtractor, PolicyModel
from .self_improvement import SelfImprovement

if TYPE_CHECKING:
    from .ml_model import ResourceModel


class Founder(Agent):
    """Agent that reads system performance metrics and outputs suggestions."""

    def __init__(
        self,
        model: Optional["ResourceModel"] = None,
        policy: Optional[PolicyModel] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
    ) -> None:
        if model is None:
            from .ml_model import ResourceModel

            model = ResourceModel()
        self.model = model
        self.policy = policy or PolicyModel()
        self.feature_extractor = feature_extractor or FeatureExtractor()

        self._self_improvement_interval = float(
            os.getenv("FOUNDER_SELF_IMPROVEMENT_INTERVAL", 3600)
        )
        self._action_logger = ActionLogger("monitoring/self_improvement.log")
        self._improver = SelfImprovement()
        self._improvement_thread = threading.Thread(
            target=self._self_improvement_loop, daemon=True
        )
        self._improvement_thread.start()

    def perform(self) -> str:
        metrics = self._collect_metrics()
        suggestions = self._generate_suggestions(metrics)
        return "\n".join(suggestions)

    def _collect_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        if psutil is not None:
            metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            metrics["memory_percent"] = psutil.virtual_memory().percent
        elif hasattr(os, "getloadavg"):
            metrics["load_avg"] = os.getloadavg()
        return metrics

    def _generate_suggestions(self, metrics: Dict[str, Any]) -> list[str]:
        predictions = self.model.predict_next()
        cpu = predictions.get("cpu_percent") or metrics.get("cpu_percent")
        mem = predictions.get("memory_percent") or metrics.get("memory_percent")
        features = self.feature_extractor.extract(
            {"cpu_percent": cpu, "memory_percent": mem}
        )
        return self.policy.predict(features)

    def plan_tool_updates(self) -> str:
        """Run the Genesis team and self-improvement routine."""

        manager = GenesisTeamManager()
        logs = manager.run()

        improver = SelfImprovement()
        results = improver.run()

        summary = [f"{agent}: {output}" for agent, output in logs.items()]
        if results["suggestions"]:
            summary.append("Self-Improvement Suggestions:")
            summary.extend(results["suggestions"])
        if results["actions"]:
            summary.append("Triggered Actions:")
            summary.extend(results["actions"])
        return "\n".join(summary)

    def _self_improvement_loop(self) -> None:
        thresholds = {"cpu_percent": 50.0, "memory_percent": 50.0}
        while True:
            time.sleep(self._self_improvement_interval)
            try:
                results = self._improver.run()
                self._action_logger.log(
                    {"suggestions": results["suggestions"], "actions": results["actions"]}
                )
                if any("fail" in a.lower() for a in results["actions"]):
                    self._improver.evaluate_and_rollback(thresholds)
            except Exception as exc:  # pragma: no cover - background loop safety
                self._action_logger.log({"error": str(exc)})
