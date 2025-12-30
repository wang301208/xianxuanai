from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import torch
from torch import nn

from .auto_gpt_rl_env import (
    AutoGPTRLEnvironment,
    AutoGPTRewardFunction,
    RolloutCollector,
)
from .auto_gpt_reward_evaluators import (
    EfficiencyRewardEvaluator,
    RuleCheckRewardEvaluator,
    UnitTestRewardEvaluator,
)
from .auto_gpt_rl_env import AutoGPTOrchestrator, RewardConfig, RewardEvaluator
from ..learning import AutoGPTPolicyHead, EpisodeRecord, ExperienceHub, PolicyConfig
from ..metrics.rl_metrics import RLMetrics, RLMetricsTracker
from .ppo import PPO, PPOConfig


def default_goal_encoder(text: str) -> torch.Tensor:
    """Encode goal text using simple bag-of-ngrams projection."""
    tokens = text.lower().split()
    vec = torch.zeros(64, dtype=torch.float32)
    for token in tokens:
        idx = hash(token) % vec.numel()
        vec[idx] += 1.0
    return vec / max(1.0, vec.norm())


class AutoGPTRLTrainer:
    """Coordinate PPO rollouts, policy updates, and experience logging."""

    def __init__(
        self,
        orchestrator: AutoGPTOrchestrator,
        experience_hub: ExperienceHub,
        metrics_tracker: RLMetricsTracker,
        policy_config: PolicyConfig,
        device: torch.device | None = None,
        reward_config: RewardConfig | None = None,
        evaluators: Iterable[RewardEvaluator] | None = None,
    ):
        self.device = device or torch.device("cpu")
        self.policy = AutoGPTPolicyHead(policy_config).to(self.device)
        self.value = nn.Sequential(
            nn.Linear(policy_config.state_dim, policy_config.hidden_dim),
            nn.ReLU(),
            nn.Linear(policy_config.hidden_dim, 1),
        ).to(self.device)
        self.env = AutoGPTRLEnvironment(
            orchestrator=orchestrator,
            reward_fn=AutoGPTRewardFunction(
                evaluators=list(
                    evaluators
                    or [
                        UnitTestRewardEvaluator(),
                        RuleCheckRewardEvaluator(),
                        EfficiencyRewardEvaluator(),
                    ]
                ),
                config=reward_config or RewardConfig(),
            ),
            goal_encoder=default_goal_encoder,
            log_path=Path("results") / "rl" / "autogpt_episode.json",
        )
        self.collector = RolloutCollector(self.env, self.policy, self.value, device=self.device)
        self.trainer = PPO(self.policy, self.value, PPOConfig())
        self.experience_hub = experience_hub
        self.metrics_tracker = metrics_tracker
        self.policy_version = "v0"

    def train(self, task_specs: Iterable[Dict[str, Any]]) -> None:
        trajectories = self.collector.collect(task_specs)
        if not trajectories:
            return
        metadata = self.collector.last_metadata
        self.trainer.update(trajectories)
        total_reward = sum(traj["return"].sum().item() for traj in trajectories)
        total_steps = sum(traj["state"].shape[0] for traj in trajectories)
        guardrail_breaches = sum(m.get("guardrail_breaches", 0) for m in metadata)
        evaluated_runs = sum(1 for m in metadata if m.get("evaluation_events", 0) > 0)
        coverage = evaluated_runs / max(1, len(metadata))
        record = {
            "task_count": len(trajectories),
            "total_reward": total_reward,
            "total_steps": total_steps,
            "guardrail_breaches": guardrail_breaches,
            "evaluation_coverage": coverage,
        }
        reward_gain = total_reward / max(1, total_steps)
        metrics = RLMetrics(
            reward_gain=reward_gain,
            guardrail_breaches=guardrail_breaches,
            evaluation_coverage=coverage,
        )
        self.metrics_tracker.record(metrics)
        self._persist_experience(trajectories, metadata, record)

    def _persist_experience(
        self,
        trajectories: Iterable[Dict[str, Any]],
        metadata: Iterable[Dict[str, Any]],
        summary: Dict[str, Any],
    ) -> None:
        for idx, (traj, meta) in enumerate(zip(trajectories, metadata)):
            path = Path("data/learning/hub") / f"trajectory_{idx}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(traj, path)
            self.experience_hub.append(
                EpisodeRecord(
                    task_id=f"autogpt-{idx}",
                    policy_version=self.policy_version,
                    total_reward=traj["return"].sum().item(),
                    steps=traj["state"].shape[0],
                    success=traj["return"].sum().item() > 0,
                    metadata={**summary, **meta},
                    trajectory_path=str(path),
                )
            )
