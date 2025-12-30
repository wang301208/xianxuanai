from BrainSimulationSystem.evaluation.developmental import (
    AssessmentMetric,
    DevelopmentalAssessment,
    DevelopmentalEvaluator,
)


class Agent:
    def __init__(self):
        self.calls = []

    def perform(self, task_name: str, payload=None):
        self.calls.append((task_name, payload))
        if task_name == "imitation_test":
            return {"accuracy": 0.8, "commands": 0.75}
        return {"score": 0.9}


def test_developmental_evaluator_checks_thresholds():
    agent = Agent()
    evaluator = DevelopmentalEvaluator(
        [
            DevelopmentalAssessment(
                name="imitate",
                stage="juvenile",
                description="",
                metrics=[AssessmentMetric("accuracy", 0.7), AssessmentMetric("commands", 0.7)],
                evaluator=lambda a: a.perform("imitation_test"),
            )
        ]
    )
    results = evaluator.run_stage(agent, "juvenile")
    assert results[0].passed is True


def test_stage_passed_requires_ratio():
    agent = Agent()
    assessments = [
        DevelopmentalAssessment(
            name="pass",
            stage="infant",
            description="",
            metrics=[AssessmentMetric("score", 0.6)],
            evaluator=lambda a: {"score": 0.8},
        ),
        DevelopmentalAssessment(
            name="fail",
            stage="infant",
            description="",
            metrics=[AssessmentMetric("score", 0.9)],
            evaluator=lambda a: {"score": 0.5},
        ),
    ]
    evaluator = DevelopmentalEvaluator(assessments)
    assert evaluator.stage_passed(agent, "infant", min_pass_rate=0.4) is True
    assert evaluator.stage_passed(agent, "infant", min_pass_rate=0.8) is False
