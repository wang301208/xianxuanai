"""Self-evolving cognition module integrating performance monitoring.

This module provides :class:`SelfEvolvingCognition` which observes performance
metrics and automatically triggers evolution of a cognitive architecture using
:class:`EvolvingCognitiveArchitecture`.  After each evolution step the
performance and architecture version are recorded, enabling rollback and
comparison of past versions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Union

try:  # pragma: no cover - support both repo-root and `modules/` on sys.path
    from modules.monitoring.collector import MetricEvent, RealTimeMetricsCollector
except ModuleNotFoundError:  # pragma: no cover
    from monitoring.collector import MetricEvent, RealTimeMetricsCollector
from .evolving_cognitive_architecture import EvolvingCognitiveArchitecture


@dataclass
class EvolutionRecord:
    """Record of a single evolution step."""

    version: int
    architecture: Dict[str, float]
    performance: float
    metrics: Dict[str, float] = field(default_factory=dict)
    confidence: Optional[float] = None
    correctness: Optional[float] = None


FeedbackSource = Union[
    Iterable[Any],
    Mapping[Any, Any],
    Callable[[MetricEvent], Any],
]


class SelfEvolvingCognition:
    """Automatically evolve a cognitive architecture based on performance metrics."""

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "throughput": 1.0,
        "latency": 1.0,
        "energy": 1.0,
        "confidence": 1.0,
        "correctness": 1.0,
    }

    def __init__(
        self,
        initial_architecture: Dict[str, float],
        evolver: EvolvingCognitiveArchitecture,
        collector: Optional[RealTimeMetricsCollector] = None,
        scoring_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.architecture = initial_architecture
        self.evolver = evolver
        self.collector = collector
        self.scoring_weights = self.DEFAULT_WEIGHTS.copy()
        if scoring_weights:
            self.scoring_weights.update({k: float(v) for k, v in scoring_weights.items()})
        self._processed_events = 0
        self.version = 0
        initial_perf = self.evolver.fitness_fn(initial_architecture)
        self.history: List[EvolutionRecord] = [
            EvolutionRecord(
                self.version,
                initial_architecture.copy(),
                initial_perf,
                {"resource_score": initial_perf},
            )
        ]

    # ------------------------------------------------------------------
    def _score_event(
        self, event: MetricEvent, feedback_metrics: Optional[Dict[str, float]] = None
    ) -> float:
        """Derive a scalar performance score from a metric event."""

        weights = self.scoring_weights
        resource_score = (
            weights.get("throughput", 1.0) * event.throughput
            - weights.get("latency", 1.0) * event.latency
            - weights.get("energy", 1.0) * event.energy
        )

        blended_score = resource_score
        if feedback_metrics:
            confidence = feedback_metrics.get("confidence")
            if confidence is not None:
                blended_score += weights.get("confidence", 1.0) * confidence
            correctness = feedback_metrics.get("correctness")
            if correctness is not None:
                blended_score += weights.get("correctness", 1.0) * correctness
            for key, value in feedback_metrics.items():
                if key in {"confidence", "correctness"}:
                    continue
                blended_score += weights.get(key, 1.0) * value

        return blended_score

    # ------------------------------------------------------------------
    def _process_event(self, event: MetricEvent, evaluation: Optional[Any] = None) -> None:
        """Process a new metric event and evolve the architecture."""

        feedback_metrics = self._extract_feedback_metrics(evaluation)
        performance = self._score_event(event, feedback_metrics)
        new_arch = self.evolver.evolve_architecture(self.architecture, performance)
        new_perf = self.evolver.fitness_fn(new_arch)
        self.version += 1
        self.architecture = new_arch
        record_metrics = {
            "resource_score": performance,
            "latency": event.latency,
            "energy": event.energy,
            "throughput": event.throughput,
        }
        if feedback_metrics:
            record_metrics.update(
                {k: v for k, v in feedback_metrics.items() if isinstance(v, float)}
            )
        self.history.append(
            EvolutionRecord(
                self.version,
                new_arch.copy(),
                new_perf,
                record_metrics,
                confidence=feedback_metrics.get("confidence") if feedback_metrics else None,
                correctness=feedback_metrics.get("correctness") if feedback_metrics else None,
            )
        )

    # ------------------------------------------------------------------
    def observe(self, feedback: Optional[FeedbackSource] = None) -> None:
        """Observe new metric events and trigger evolution steps."""

        if self.collector is None:
            return
        events = self.collector.events()
        new_events = events[self._processed_events :]
        feedback_iter: Optional[Iterator[Any]] = None
        if feedback is not None and not callable(feedback) and not isinstance(feedback, Mapping):
            feedback_iter = iter(feedback)

        for event in new_events:
            evaluation: Any = None
            if feedback is not None:
                if callable(feedback):
                    evaluation = feedback(event)
                elif isinstance(feedback, Mapping):
                    evaluation = feedback.get(event.stage or event.module)
                    if evaluation is None:
                        evaluation = feedback.get(event.module)
                    if evaluation is None:
                        evaluation = feedback.get(event.timestamp)
                    if evaluation is None:
                        evaluation = feedback.get(str(event.timestamp))
                elif feedback_iter is not None:
                    try:
                        evaluation = next(feedback_iter)
                    except StopIteration:
                        feedback_iter = None
            self._process_event(event, evaluation)
        self._processed_events = len(events)

    # ------------------------------------------------------------------
    def _extract_feedback_metrics(self, feedback: Optional[Any]) -> Dict[str, float]:
        """Normalise task-evaluation feedback into a metrics dictionary."""

        if feedback is None:
            return {}

        if isinstance(feedback, bool):
            return {"correctness": 1.0 if feedback else 0.0}

        if isinstance(feedback, (int, float)):
            return {"correctness": float(feedback)}

        if isinstance(feedback, Mapping):
            metrics: Dict[str, float] = {}
            for key, value in feedback.items():
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
            return metrics

        metrics: Dict[str, float] = {}
        for attr in ("confidence", "correctness", "score", "success"):
            value = getattr(feedback, attr, None)
            if isinstance(value, (int, float)):
                key = attr
                if attr == "score":
                    key = "confidence"
                if attr == "success":
                    key = "correctness"
                metrics[key] = float(value)
        return metrics

    # ------------------------------------------------------------------
    @staticmethod
    def feedback_from_event(event: MetricEvent) -> Dict[str, float]:
        """Adapt a ``MetricEvent`` with outcome metadata into feedback metrics."""

        feedback: Dict[str, float] = {}

        status = event.status.lower() if isinstance(event.status, str) else None
        correctness: Optional[float] = None
        if status is not None:
            if status in {"success", "succeeded", "passed", "ok", "complete", "completed"}:
                correctness = 1.0
            elif status in {"failure", "failed", "error", "exception"}:
                correctness = 0.0
        elif event.prediction is not None and event.actual is not None:
            correctness = 1.0 if event.prediction == event.actual else 0.0

        if correctness is not None:
            feedback["correctness"] = correctness

        if event.confidence is not None:
            feedback["confidence"] = float(event.confidence)

        meta = getattr(event, "metadata", None)
        if isinstance(meta, Mapping):
            if "correctness" not in feedback and "success" in meta:
                try:
                    feedback["correctness"] = 1.0 if bool(meta.get("success")) else 0.0
                except Exception:
                    pass
            if "confidence" not in feedback:
                value = meta.get("rating", meta.get("score", meta.get("satisfaction")))
                if isinstance(value, (int, float)):
                    raw = float(value)
                    if raw <= 1.0:
                        feedback["confidence"] = max(0.0, min(1.0, raw))
                    elif raw <= 5.0:
                        feedback["confidence"] = max(0.0, min(1.0, raw / 5.0))
                    else:
                        feedback["confidence"] = max(0.0, min(1.0, raw / 10.0))

        return feedback

    # ------------------------------------------------------------------
    def rollback(self, version: int) -> Dict[str, float]:
        """Rollback to a previous architecture version."""

        for record in self.history:
            if record.version == version:
                self.architecture = record.architecture.copy()
                self.version = record.version
                return self.architecture
        raise ValueError(f"Version {version} not found in history")

    # ------------------------------------------------------------------
    def compare(self, v1: int, v2: int) -> Dict[str, float]:
        """Compare the performance between two versions."""

        rec1 = next(r for r in self.history if r.version == v1)
        rec2 = next(r for r in self.history if r.version == v2)
        return {"performance_diff": rec2.performance - rec1.performance}
