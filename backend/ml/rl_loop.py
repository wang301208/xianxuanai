"""Orchestrate hierarchical reinforcement learning with curiosity signals."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from BrainSimulationSystem.environment.base import EnvironmentController, PerceptionPacket
from BrainSimulationSystem.environment.policy_bridge import HierarchicalPolicyBridge, HighLevelDecision
from backend.ml.experience_collector import ActiveCuriositySelector
try:
    from BrainSimulationSystem.motivation.curiosity import SocialCuriosityEngine
    from BrainSimulationSystem.motivation.curiosity_bridge import CuriosityStimulusEncoder
except Exception:  # pragma: no cover - optional dependency chain
    SocialCuriosityEngine = None  # type: ignore[assignment]
    CuriosityStimulusEncoder = None  # type: ignore[assignment]
try:
    from modules.brain import SelfLearningBrain
except Exception:  # pragma: no cover - fallback when brain stack unavailable
    class SelfLearningBrain:  # type: ignore[override]
        def curiosity_driven_learning(self, _sample):
            return {}

try:
    from modules.learning import EpisodeRecord, ExperienceHub
except Exception:  # pragma: no cover - fallback minimal implementations
    from modules.learning.experience_hub import EpisodeRecord, ExperienceHub


class SupportsLearningAgent(Protocol):
    """Protocol implemented by deep RL agents used in the loop."""

    def select_action(self, state: np.ndarray) -> Tuple[int, Dict[str, float]]: ...

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None: ...

    def update(self) -> Optional[Dict[str, float]]: ...


PolicyTrainerFn = Callable[[Iterable[Dict[str, Any]]], Any]


@dataclass
class ReinforcementTrainerConfig:
    """Configuration for the reinforcement learning loop."""

    max_steps_per_episode: int = 256
    policy_version: str = "rl-stage-0"
    task_id: str = "default-task"
    train_every_episodes: int = 1
    min_reward_for_hub: float = -float("inf")
    log_curiosity_samples: bool = True
    curiosity_weight: float = 0.1
    curiosity_decay: float = 1.0


class MemoryBridge(Protocol):
    def record_episode(
        self,
        episode: EpisodeRecord,
        *,
        metrics: Optional[Dict[str, Any]] = None,
        curiosity_samples: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Any: ...


class ReinforcementLearningLoop:
    """Couple BrainSimulation decisions, RL policies, curiosity, and storage."""

    def __init__(
        self,
        *,
        env: EnvironmentController,
        policy_bridge: HierarchicalPolicyBridge,
        rl_agent: SupportsLearningAgent,
        experience_hub: ExperienceHub | None = None,
        policy_trainer: PolicyTrainerFn | None = None,
        curiosity_selector: ActiveCuriositySelector | None = None,
        self_learning_brain: SelfLearningBrain | None = None,
        config: ReinforcementTrainerConfig | None = None,
        agent_id: str = "rl-agent",
        curiosity_engine: Optional[SocialCuriosityEngine] = None,
        curiosity_encoder: Optional[CuriosityStimulusEncoder] = None,
        memory_bridge: MemoryBridge | None = None,
    ) -> None:
        self.env = env
        self.policy_bridge = policy_bridge
        self.rl_agent = rl_agent
        self.config = config or ReinforcementTrainerConfig()
        self.agent_id = agent_id
        self.experience_hub = experience_hub or ExperienceHub(Path("data/experience/rl"))
        self.policy_trainer = policy_trainer
        self.curiosity_selector = curiosity_selector or ActiveCuriositySelector()
        self.self_learning_brain = self_learning_brain or SelfLearningBrain()
        self._episode_counter = 0
        self._memory_bridge = memory_bridge
        self.curiosity_engine = curiosity_engine
        self.curiosity_encoder = curiosity_encoder
        if self.curiosity_engine is not None and self.curiosity_encoder is None and CuriosityStimulusEncoder is not None:
            self.curiosity_encoder = CuriosityStimulusEncoder()
        self._intrinsic_weight = max(0.0, (self.config.curiosity_weight))
        self._curiosity_decay = max(0.0, self.config.curiosity_decay)

    # ------------------------------------------------------------------ #
    def run_episode(self, *, task_id: str | None = None) -> EpisodeRecord:
        """Execute a single rollout and persist metadata."""

        packet = self.env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        curiosity_samples: List[Dict[str, Any]] = []

        last_metrics: Optional[Dict[str, float]] = None
        while not done and steps < self.config.max_steps_per_episode:
            action, action_idx, obs_vector, decision, _ = self.policy_bridge.select_action(
                packet, self.rl_agent
            )
            next_packet, external_reward, done, info = self.env.step(action)
            intrinsic_reward, stimulus = self._compute_curiosity_reward(next_packet, info)
            reward = external_reward + intrinsic_reward
            next_decision = self.policy_bridge.decision_from_packet(next_packet)
            next_obs_vector = self.policy_bridge.encode(next_packet, next_decision)

            self.rl_agent.observe(obs_vector, action_idx, reward, next_obs_vector, done)
            metrics = self.rl_agent.update()
            if metrics:
                last_metrics = metrics
            total_reward += float(reward)
            steps += 1

            sample = self._build_sample(
                packet,
                reward,
                info,
                decision,
                stimulus=stimulus,
                intrinsic_reward=intrinsic_reward,
                external_reward=external_reward,
            )
            if self.curiosity_selector.consider(sample):
                curiosity_samples.append(sample)
            self.self_learning_brain.curiosity_driven_learning(sample)
            packet = next_packet

        episode = EpisodeRecord(
            task_id=task_id or self.config.task_id,
            policy_version=self.config.policy_version,
            total_reward=float(total_reward),
            steps=steps,
            success=bool(total_reward > 0.0),
            metadata={
                "timestamp": time.time(),
                "agent_metrics": last_metrics or {},
            },
        )
        if episode.total_reward >= self.config.min_reward_for_hub:
            self.experience_hub.append(episode)

        self._episode_counter += 1
        if curiosity_samples and self.policy_trainer and (
            self._episode_counter % max(1, self.config.train_every_episodes) == 0
        ):
            self.policy_trainer(curiosity_samples)

        if self._memory_bridge is not None:
            samples = curiosity_samples if self.config.log_curiosity_samples else None
            self._memory_bridge.record_episode(
                episode,
                metrics=episode.metadata.get("agent_metrics"),
                curiosity_samples=samples,
            )

        return episode

    # ------------------------------------------------------------------ #
    def _build_sample(
        self,
        packet: PerceptionPacket,
        reward: float,
        info: Dict[str, Any],
        decision: HighLevelDecision,
        *,
        stimulus: Optional[Dict[str, Any]] = None,
        intrinsic_reward: float = 0.0,
        external_reward: float | None = None,
    ) -> Dict[str, Any]:
        usage = info.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        sample = {
            "state": decision.intent,
            "reward": float(reward),
            "agent_id": self.agent_id,
            "usage": {k: float(v) for k, v in usage.items() if isinstance(v, (int, float))},
            "confidence": decision.confidence,
            "metadata": decision.metadata,
        }
        if packet.state_vector:
            sample["state_vector"] = list(packet.state_vector)
        meta = sample["metadata"]
        if stimulus:
            meta["novelty"] = stimulus.get("novelty", meta.get("novelty", 0.0))
            meta["complexity"] = stimulus.get("complexity", meta.get("complexity", 0.0))
            meta["social_context"] = stimulus.get("social_context", False)
            meta["curiosity_reward"] = intrinsic_reward
        if external_reward is not None:
            meta["external_reward"] = external_reward
        return sample

    # ------------------------------------------------------------------ #
    def _compute_curiosity_reward(
        self,
        packet: PerceptionPacket,
        info: Dict[str, Any],
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        if (
            self.curiosity_engine is None
            or self.curiosity_encoder is None
            or self._intrinsic_weight <= 0.0
        ):
            return 0.0, None

        stimulus = self.curiosity_encoder.build(packet, info=info, metadata=packet.metadata)
        raw_curiosity = float(self.curiosity_engine.compute_integrated_curiosity(stimulus))
        intrinsic = self._intrinsic_weight * raw_curiosity
        social_feedback = info.get("social_feedback")
        if isinstance(social_feedback, Mapping):
            reward_signal = float(social_feedback.get("reward", 0.0))
            social_type = str(social_feedback.get("type", "face"))
            try:
                self.curiosity_engine.update_social_parameters(reward_signal, social_type)
            except Exception:  # pragma: no cover - defensive
                pass
        self._intrinsic_weight *= max(0.0, self._curiosity_decay)
        return intrinsic, stimulus


__all__ = [
    "ReinforcementLearningLoop",
    "ReinforcementTrainerConfig",
]
