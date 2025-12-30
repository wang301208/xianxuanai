from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .auto_gpt_rl_env import RewardBreakdown, RewardConfig, RewardEvaluator


@dataclass
class UnitTestRewardEvaluator(RewardEvaluator):
    """Assign reward based on unit-test pass rate for synthetic tasks."""

    success_threshold: float = 0.95

    def evaluate(self, signal: Dict, config: RewardConfig) -> RewardBreakdown:
        breakdown = RewardBreakdown()
        test_results = signal.get("unit_test_results", {})
        total = test_results.get("total", 0)
        passed = test_results.get("passed", 0)
        if total:
            rate = passed / total
            if rate >= self.success_threshold:
                breakdown.success = config.success_bonus
            else:
                breakdown.success = config.success_bonus * rate
        return breakdown


@dataclass
class RuleCheckRewardEvaluator(RewardEvaluator):
    """Penalise loops, contradictions, and policy violations detected by rules."""

    loop_limit: int = 3

    def evaluate(self, signal: Dict, config: RewardConfig) -> RewardBreakdown:
        breakdown = RewardBreakdown()
        loops = signal.get("loop_count", 0)
        if loops > self.loop_limit:
            breakdown.loop = config.loop_penalty * (loops - self.loop_limit)
        contradiction = signal.get("contradiction", False)
        if contradiction:
            breakdown.contradiction = config.contradiction_penalty
        timeout = signal.get("timed_out", False)
        if timeout:
            breakdown.timeout = config.timeout_penalty
        return breakdown


@dataclass
class EfficiencyRewardEvaluator(RewardEvaluator):
    """Favour efficient completion measured by plan progress per step."""

    expected_steps: int = 20

    def evaluate(self, signal: Dict, config: RewardConfig) -> RewardBreakdown:
        breakdown = RewardBreakdown()
        progress = signal.get("plan_progress", 0.0)
        steps = max(1, signal.get("step", 1))
        efficiency = progress / steps
        baseline = 1.0 / self.expected_steps
        delta = efficiency - baseline
        breakdown.efficiency = config.efficiency_weight * delta
        return breakdown
