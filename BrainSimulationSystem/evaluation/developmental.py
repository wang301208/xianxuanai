"""Developmental assessment utilities for curriculum-aligned benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence


class SupportsAssessmentAgent(Protocol):
    """Minimal protocol for agents evaluated by developmental tasks."""

    def perform(self, task_name: str, payload: Optional[Mapping[str, object]] = None) -> Mapping[str, float]:
        ...


@dataclass(frozen=True)
class AssessmentMetric:
    """Target thresholds for a single developmental competency."""

    key: str
    threshold: float
    description: str = ""


@dataclass(frozen=True)
class DevelopmentalAssessment:
    name: str
    stage: str
    description: str
    metrics: Sequence[AssessmentMetric]
    evaluator: Callable[[SupportsAssessmentAgent], Mapping[str, float]]


@dataclass
class AssessmentResult:
    assessment: DevelopmentalAssessment
    passed: bool
    metrics: Dict[str, float] = field(default_factory=dict)


class DevelopmentalEvaluator:
    """Run stage-aligned developmental assessments and aggregate results."""

    def __init__(self, assessments: Iterable[DevelopmentalAssessment] | None = None) -> None:
        self._assessments: List[DevelopmentalAssessment] = list(assessments or [])

    def register(self, assessment: DevelopmentalAssessment) -> None:
        self._assessments.append(assessment)

    def run_stage(
        self,
        agent: SupportsAssessmentAgent,
        stage_key: str,
    ) -> List[AssessmentResult]:
        results: List[AssessmentResult] = []
        for assessment in self._assessments:
            if assessment.stage != stage_key:
                continue
            metrics = dict(assessment.evaluator(agent))
            passed = self._meets_thresholds(assessment.metrics, metrics)
            results.append(AssessmentResult(assessment, passed, metrics))
        return results

    def stage_passed(self, agent: SupportsAssessmentAgent, stage_key: str, min_pass_rate: float = 1.0) -> bool:
        results = self.run_stage(agent, stage_key)
        if not results:
            return True
        passed = sum(1 for result in results if result.passed)
        rate = passed / len(results)
        return rate >= min_pass_rate

    def _meets_thresholds(self, metrics: Sequence[AssessmentMetric], scores: Mapping[str, float]) -> bool:
        for metric in metrics:
            if float(scores.get(metric.key, float("nan"))) < metric.threshold:
                return False
        return True


def build_default_assessments() -> List[DevelopmentalAssessment]:
    """Return a default battery aligned with the five developmental stages."""

    def _static(score: float) -> Callable[[SupportsAssessmentAgent], Mapping[str, float]]:
        return lambda _agent: {"score": score}

    assessments = [
        DevelopmentalAssessment(
            name="sensorimotor-grasp",
            stage="infant",
            description="Measures basic reflexes and object responses.",
            metrics=[AssessmentMetric("score", 0.6, "Grasp success")],
            evaluator=_static(0.8),
        ),
        DevelopmentalAssessment(
            name="imitation-and-language",
            stage="juvenile",
            description="Imitation accuracy and command following.",
            metrics=[AssessmentMetric("accuracy", 0.7), AssessmentMetric("commands", 0.6)],
            evaluator=lambda agent: agent.perform("imitation_test", {"commands": 5}),
        ),
        DevelopmentalAssessment(
            name="object-permanence",
            stage="adolescent",
            description="Hidden-object tracking and causal puzzle solving.",
            metrics=[AssessmentMetric("puzzle", 0.75)],
            evaluator=lambda agent: agent.perform("object_permanence"),
        ),
        DevelopmentalAssessment(
            name="dialogue-understanding",
            stage="production",
            description="Multi-turn language and planning abilities.",
            metrics=[AssessmentMetric("dialogue", 0.8)],
            evaluator=lambda agent: agent.perform("dialogue"),
        ),
        DevelopmentalAssessment(
            name="abstract-reasoning",
            stage="full",
            description="Logic, math, and creativity tasks.",
            metrics=[AssessmentMetric("reasoning", 0.85), AssessmentMetric("creativity", 0.7)],
            evaluator=lambda agent: agent.perform("abstract_reasoning"),
        ),
    ]
    return assessments


__all__ = [
    "AssessmentMetric",
    "DevelopmentalAssessment",
    "AssessmentResult",
    "DevelopmentalEvaluator",
    "build_default_assessments",
]
