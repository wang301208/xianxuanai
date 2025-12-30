from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Protocol, Tuple

import torch
from torch import Tensor, nn


@dataclass
class AutoGPTState:
    """State representation for an AutoGPT task episode."""

    step: int
    max_steps: int
    goal_embedding: Tensor
    plan_progress: float
    tool_usage_count: int
    last_reward: float
    confidence: float
    stalled_steps: int

    def to_tensor(self) -> Tensor:
        scalars = torch.tensor(
            [
                self.step / max(1, self.max_steps),
                self.plan_progress,
                float(self.tool_usage_count),
                self.last_reward,
                self.confidence,
                float(self.stalled_steps),
            ],
            dtype=torch.float32,
        )
        return torch.cat([self.goal_embedding, scalars], dim=0)


@dataclass
class AutoGPTAction:
    """Action representation controlling high-level behaviour."""

    action_logits: Tensor
    description: str

    def sample(self) -> Tensor:
        dist = torch.distributions.Categorical(logits=self.action_logits)
        return dist.sample()


class AutoGPTOrchestrator(Protocol):
    """Protocol for orchestrating AutoGPT task execution."""

    def reset(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def step(self, policy_directive: Dict[str, Any]) -> Dict[str, Any]:
        ...


@dataclass
class RewardConfig:
    success_bonus: float = 5.0
    loop_penalty: float = -1.0
    efficiency_weight: float = 1.0
    contradiction_penalty: float = -0.5
    timeout_penalty: float = -2.0


@dataclass
class RewardBreakdown:
    success: float = 0.0
    loop: float = 0.0
    efficiency: float = 0.0
    contradiction: float = 0.0
    timeout: float = 0.0

    def total(self) -> float:
        return self.success + self.loop + self.efficiency + self.contradiction + self.timeout


class RewardEvaluator(Protocol):
    def evaluate(self, signal: Dict[str, Any], config: RewardConfig) -> RewardBreakdown:
        ...


@dataclass
class AutoGPTRewardFunction:
    evaluators: List[RewardEvaluator]
    config: RewardConfig = field(default_factory=RewardConfig)

    def __call__(self, signal: Dict[str, Any]) -> Tuple[float, RewardBreakdown]:
        breakdown = RewardBreakdown()
        for evaluator in self.evaluators:
            result = evaluator.evaluate(signal, self.config)
            breakdown.success += result.success
            breakdown.loop += result.loop
            breakdown.efficiency += result.efficiency
            breakdown.contradiction += result.contradiction
            breakdown.timeout += result.timeout
        return breakdown.total(), breakdown


class AutoGPTRLEnvironment:
    """Environment wrapper providing RL-friendly interface to AutoGPT runs."""

    def __init__(
        self,
        orchestrator: AutoGPTOrchestrator,
        reward_fn: AutoGPTRewardFunction,
        goal_encoder: Callable[[str], Tensor],
        log_path: Path | None = None,
    ):
        self.orchestrator = orchestrator
        self.reward_fn = reward_fn
        self.goal_encoder = goal_encoder
        self.log_path = log_path
        self.current_state: AutoGPTState | None = None
        self.episode_log: List[Dict[str, Any]] = []

    def reset(self, task_spec: Dict[str, Any]) -> AutoGPTState:
        response = self.orchestrator.reset(task_spec)
        goal_embedding = self.goal_encoder(response["goal_text"])
        self.current_state = AutoGPTState(
            step=0,
            max_steps=response.get("max_steps", 50),
            goal_embedding=goal_embedding,
            plan_progress=response.get("plan_progress", 0.0),
            tool_usage_count=0,
            last_reward=0.0,
            confidence=response.get("confidence", 0.5),
            stalled_steps=0,
        )
        self.episode_log = [self._log_entry(response, reward=0.0, breakdown=RewardBreakdown())]
        return self.current_state

    def step(self, action: Tensor) -> Tuple[AutoGPTState, float, bool, Dict[str, Any]]:
        if self.current_state is None:
            raise RuntimeError("Environment must be reset before stepping.")

        directive = {"action": int(action.item())}
        response = self.orchestrator.step(directive)
        reward, breakdown = self.reward_fn(response)

        tool_count = response.get("tool_usage_count", self.current_state.tool_usage_count)
        stalled = response.get("stalled_steps", self.current_state.stalled_steps)
        self.current_state = AutoGPTState(
            step=response.get("step", self.current_state.step + 1),
            max_steps=response.get("max_steps", self.current_state.max_steps),
            goal_embedding=self.current_state.goal_embedding,
            plan_progress=response.get("plan_progress", self.current_state.plan_progress),
            tool_usage_count=tool_count,
            last_reward=reward,
            confidence=response.get("confidence", self.current_state.confidence),
            stalled_steps=stalled,
        )

        done = bool(response.get("done", False))
        info = {"breakdown": breakdown, "raw": response}
        self.episode_log.append(self._log_entry(response, reward=reward, breakdown=breakdown))

        if done:
            self._persist_episode()
        return self.current_state, reward, done, info

    def _log_entry(self, response: Dict[str, Any], reward: float, breakdown: RewardBreakdown) -> Dict[str, Any]:
        entry = {
            "step": response.get("step"),
            "action": response.get("action"),
            "observation": response.get("observation"),
            "reward": reward,
            "breakdown": {
                "success": breakdown.success,
                "loop": breakdown.loop,
                "efficiency": breakdown.efficiency,
                "contradiction": breakdown.contradiction,
                "timeout": breakdown.timeout,
            },
            "done": response.get("done", False),
            "confidence": response.get("confidence"),
            "latency": response.get("latency"),
            "raw": response,
        }
        return entry

    def _persist_episode(self) -> None:
        if not self.log_path:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", encoding="utf-8") as f:
            json.dump(self.episode_log, f, ensure_ascii=False, indent=2)


class RolloutCollector:
    """Collects trajectories compatible with PPO/SAC trainers."""

    def __init__(
        self,
        env: AutoGPTRLEnvironment,
        policy: nn.Module,
        value_network: nn.Module,
        discount: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device | None = None,
    ):
        self.env = env
        self.policy = policy
        self.value_network = value_network
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.device = device or torch.device("cpu")
        self.last_metadata: List[Dict[str, Any]] = []

    def collect(self, task_specs: Iterable[Dict[str, Any]]) -> List[Dict[str, Tensor]]:
        trajectories: List[Dict[str, Tensor]] = []
        self.last_metadata = []
        for spec in task_specs:
            state = self.env.reset(spec)
            states: List[Tensor] = []
            actions: List[Tensor] = []
            logps: List[Tensor] = []
            rewards: List[float] = []
            values: List[Tensor] = []

            done = False
            while not done:
                state_tensor = state.to_tensor().to(self.device)
                dist = self.policy(state_tensor)
                action = dist.sample()
                logp = dist.log_prob(action)
                value = self.value_network(state_tensor.unsqueeze(0))

                next_state, reward, done, _info = self.env.step(action)
                states.append(state_tensor)
                actions.append(action)
                logps.append(logp)
                rewards.append(reward)
                values.append(value)
                state = next_state

            metadata = self._summarize_episode(self.env.episode_log)
            trajectory = self._build_trajectory(states, actions, logps, rewards, values)
            trajectories.append(trajectory)
            self.last_metadata.append(metadata)
        return trajectories

    def _build_trajectory(
        self,
        states: List[Tensor],
        actions: List[Tensor],
        logps: List[Tensor],
        rewards: List[float],
        values: List[Tensor],
    ) -> Dict[str, Tensor]:
        returns = []
        advs = []
        gae = 0.0
        next_value = 0.0
        for reward, value in zip(reversed(rewards), reversed(values)):
            delta = reward + self.discount * next_value - value.item()
            gae = delta + self.discount * self.gae_lambda * gae
            advs.insert(0, gae)
            next_value = value.item()
        returns = torch.tensor(
            [adv + val.item() for adv, val in zip(advs, values)],
            dtype=torch.float32,
            device=self.device,
        )
        trajectory = {
            "state": torch.stack(states).to(self.device),
            "action": torch.stack(actions).to(self.device),
            "logp": torch.stack(logps).to(self.device),
            "return": returns,
            "adv": torch.tensor(advs, dtype=torch.float32, device=self.device),
        }
        return trajectory

    def _summarize_episode(self, log: List[Dict[str, Any]]) -> Dict[str, Any]:
        guardrail = sum(1 for entry in log if entry.get("raw", {}).get("guardrail_breach"))
        evaluations = sum(1 for entry in log if entry.get("raw", {}).get("evaluation", False))
        return {
            "steps": len(log),
            "guardrail_breaches": guardrail,
            "evaluation_events": evaluations,
        }
