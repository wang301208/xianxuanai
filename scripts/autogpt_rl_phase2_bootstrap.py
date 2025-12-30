from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import torch

from modules.evolution.auto_gpt_rl_trainer import AutoGPTRLTrainer
from modules.evolution.auto_gpt_rl_env import RewardConfig
from modules.evolution.autogpt_task_orchestrator import CallbackOrchestrator
from modules.learning import ExperienceHub, PolicyConfig
from modules.metrics.rl_metrics import RLMetricsTracker


class SyntheticAutoGPTSimulator:
    """Lightweight simulator that mimics AutoGPT step/observation responses."""

    def __init__(self) -> None:
        self.goal_text = ""
        self.step_count = 0
        self.max_steps = 10
        self.plan_progress = 0.0
        self.loop_count = 0

    def reset(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        self.goal_text = task_spec["goal_text"]
        self.step_count = 0
        self.plan_progress = 0.0
        self.loop_count = 0
        return {
            "goal_text": self.goal_text,
            "step": self.step_count,
            "max_steps": self.max_steps,
            "plan_progress": self.plan_progress,
            "confidence": 0.6,
        }

    def step(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        action = directive["action"]
        self.step_count += 1
        self.plan_progress += random.random() * 0.2
        self.plan_progress = min(self.plan_progress, 1.0)
        success = self.plan_progress >= 1.0 or self.step_count >= self.max_steps
        loop = action == 0 and random.random() < 0.2
        self.loop_count = self.loop_count + 1 if loop else 0
        response = {
            "step": self.step_count,
            "max_steps": self.max_steps,
            "plan_progress": self.plan_progress,
            "confidence": random.uniform(0.4, 0.9),
            "tool_usage_count": random.randint(0, 3),
            "loop_count": self.loop_count,
            "unit_test_results": {"total": 4, "passed": int(self.plan_progress * 4)},
            "done": success,
            "observation": {"summary": "synthetic"},
            "evaluation": True,
            "guardrail_breach": False,
        }
        return response


def main() -> None:
    simulator = SyntheticAutoGPTSimulator()
    orchestrator = CallbackOrchestrator(simulator.reset, simulator.step)
    experience_hub = ExperienceHub(Path("data/learning/hub"))
    metrics_tracker = RLMetricsTracker(Path("logs/events/rl_metrics.jsonl"))
    state_dim = 64 + 6  # embedding + scalar features
    trainer = AutoGPTRLTrainer(
        orchestrator=orchestrator,
        experience_hub=experience_hub,
        metrics_tracker=metrics_tracker,
        policy_config=PolicyConfig(state_dim=state_dim, action_dim=4),
        reward_config=RewardConfig(success_bonus=5.0, loop_penalty=-1.0, efficiency_weight=1.5),
    )
    tasks = [{"goal_text": "Refactor module for better logging"}, {"goal_text": "Draft release notes"}]
    trainer.train(tasks)
    print("Latest metrics:", metrics_tracker.load()[-1])


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    main()
