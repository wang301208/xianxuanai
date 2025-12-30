"""Tests for StrategyAdjuster."""

from modules.evolution import StrategyAdjuster
from modules.monitoring.performance_diagnoser import DiagnosticIssue


def test_strategy_adjuster_applies_actions():
    issues = [
        DiagnosticIssue(kind="low_success_rate", metric="success_rate", value=0.4, threshold=0.7),
        DiagnosticIssue(kind="high_latency", metric="latency", value=1.2, threshold=0.5, module="planner"),
    ]

    adjuster = StrategyAdjuster(lr_bounds=(1e-4, 1.0), exploration_bounds=(0.0, 1.0))
    result = adjuster.propose(issues, current_params={"policy_learning_rate": 0.1})

    updates = result["updates"]
    actions = {action.parameter for action in result["actions"]}

    assert updates["policy_exploration_rate"] > 0.1
    assert updates["planner_structured_flag"] == 1.0
    assert "policy_learning_rate" in actions or "planner_structured_flag" in actions
