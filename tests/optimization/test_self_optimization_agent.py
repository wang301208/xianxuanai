"""Tests for SelfOptimizationAgent."""

from modules.optimization import SelfOptimizationAgent


def test_self_optimization_agent_updates_q_values():
    agent = SelfOptimizationAgent(actions=["increase_lr", "decrease_lr"], epsilon=0.0)
    action = agent.select_action()
    assert action in {"increase_lr", "decrease_lr"}
    result = agent.update(reward=1.0)
    assert result.action == action
    assert agent.q_values[action] > 0.0
