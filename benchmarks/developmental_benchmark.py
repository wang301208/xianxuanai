"""Run developmental stage benchmarks for the cognitive agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping

from BrainSimulationSystem.curriculum import DEFAULT_CURRICULUM, CurriculumManager
from BrainSimulationSystem.evaluation.developmental import (
    DevelopmentalEvaluator,
    build_default_assessments,
)


class DummyAgent:
    """Placeholder agent that returns canned scores for demo purposes."""

    def __init__(self, stage_scores: Mapping[str, Mapping[str, float]]) -> None:
        self.stage_scores: Dict[str, Dict[str, float]] = {
            key: dict(value) for key, value in stage_scores.items()
        }

    def perform(self, task_name: str, payload=None):
        stage = payload.get("stage") if isinstance(payload, dict) else None
        if stage and stage in self.stage_scores:
            return self.stage_scores[stage]
        return {"score": 0.5, "accuracy": 0.5, "commands": 0.5}


def run_benchmark(output: Path, agent: DummyAgent) -> None:
    evaluator = DevelopmentalEvaluator(build_default_assessments())
    curriculum = CurriculumManager(DEFAULT_CURRICULUM, evaluator=evaluator)
    results = {}

    for stage in DEFAULT_CURRICULUM:
        stage_results = []
        for assessment in evaluator.run_stage(agent, stage.stage_key):
            stage_results.append(
                {
                    "assessment": assessment.assessment.name,
                    "passed": assessment.passed,
                    "metrics": assessment.metrics,
                }
            )
            curriculum.record_assessment(
                passed=assessment.passed,
                stage_key=stage.stage_key,
                metrics=assessment.metrics,
            )
        results[stage.stage_key] = stage_results

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved developmental benchmark to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run developmental benchmarks.")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/developmental.json"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    agent = DummyAgent(
        {
            "infant": {"score": 0.7},
            "juvenile": {"accuracy": 0.8, "commands": 0.7},
            "adolescent": {"puzzle": 0.78},
            "production": {"dialogue": 0.85},
            "full": {"reasoning": 0.9, "creativity": 0.75},
        }
    )
    run_benchmark(args.output, agent)
