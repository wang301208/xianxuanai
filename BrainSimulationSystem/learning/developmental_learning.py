"""Stage-aware learning utilities for progressive RL + imitation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from BrainSimulationSystem.core.stage_manager import CurriculumStageManager
from BrainSimulationSystem.environment.interaction_loop import (
    ImitationReplayBuffer,
    InteractiveLanguageLoop,
    InteractionTransition,
    TeacherSignal,
)


@dataclass(frozen=True)
class InteractiveLoopConfig:
    mentor_interval: int = 0
    max_steps: int = 32
    language_goal: Optional[str] = None


@dataclass(frozen=True)
class MentorConfig:
    enabled: bool = False
    reward_bonus: float = 0.0
    utterance: Optional[str] = None


@dataclass(frozen=True)
class RewardShapingConfig:
    enabled: bool = True
    mentor_bonus: float = 0.05
    critique_penalty: float = 0.05
    low_confidence_penalty: float = 0.02
    reward_channel_weight: float = 0.1
    success_bonus: float = 0.0


@dataclass(frozen=True)
class OfflineTrainingConfig:
    enabled: bool = False
    algorithm: str = "dqn"
    train_every_episodes: int = 1
    batch_size: int = 32
    updates: int = 4
    mentor_fraction: float = 0.3


@dataclass(frozen=True)
class StageLearningConfig:
    stage: str
    interactive: InteractiveLoopConfig
    mentor: MentorConfig
    shaping: RewardShapingConfig
    replay_capacity: int = 2_048
    offline: OfflineTrainingConfig = OfflineTrainingConfig()


def _nested(mapping: Mapping[str, Any], key: str) -> Dict[str, Any]:
    value = mapping.get(key, {})
    return dict(value) if isinstance(value, dict) else {}


def extract_stage_learning_config(stage_config: Mapping[str, Any]) -> StageLearningConfig:
    meta = _nested(stage_config, "metadata")
    stage = str(meta.get("stage") or stage_config.get("stage") or "unknown")

    learning = _nested(stage_config, "learning")
    interactive = _nested(learning, "interactive_language_loop")
    mentor = _nested(learning, "mentor")
    shaping = _nested(learning, "reward_shaping")
    replay = _nested(learning, "replay_buffer")
    offline = _nested(learning, "offline_training")

    try:
        mentor_interval = int(interactive.get("mentor_interval", 0))
    except Exception:
        mentor_interval = 0
    mentor_interval = max(0, mentor_interval)

    try:
        max_steps = int(interactive.get("max_steps", 32))
    except Exception:
        max_steps = 32
    max_steps = max(1, max_steps)

    language_goal = interactive.get("language_goal")
    language_goal = str(language_goal) if isinstance(language_goal, str) and language_goal.strip() else None

    mentor_enabled = mentor.get("enabled")
    mentor_enabled = bool(mentor_enabled) if mentor_enabled is not None else False
    try:
        mentor_bonus = float(mentor.get("reward_bonus", 0.0))
    except Exception:
        mentor_bonus = 0.0
    utterance = mentor.get("utterance")
    utterance = str(utterance) if isinstance(utterance, str) and utterance.strip() else None

    shaping_enabled = shaping.get("enabled")
    shaping_enabled = bool(shaping_enabled) if shaping_enabled is not None else True

    def _float(key: str, default: float) -> float:
        try:
            return float(shaping.get(key, default))
        except Exception:
            return float(default)

    shaping_cfg = RewardShapingConfig(
        enabled=shaping_enabled,
        mentor_bonus=_float("mentor_bonus", 0.05),
        critique_penalty=_float("critique_penalty", 0.05),
        low_confidence_penalty=_float("low_confidence_penalty", 0.02),
        reward_channel_weight=_float("reward_channel_weight", 0.1),
        success_bonus=_float("success_bonus", 0.0),
    )

    try:
        replay_capacity = int(replay.get("capacity", 2_048))
    except Exception:
        replay_capacity = 2_048
    replay_capacity = max(1, replay_capacity)

    offline_enabled = offline.get("enabled")
    offline_enabled = bool(offline_enabled) if offline_enabled is not None else False
    try:
        train_every = int(offline.get("train_every_episodes", 1))
    except Exception:
        train_every = 1
    train_every = max(1, train_every)
    try:
        batch_size = int(offline.get("batch_size", 32))
    except Exception:
        batch_size = 32
    batch_size = max(1, batch_size)
    try:
        updates = int(offline.get("updates", 4))
    except Exception:
        updates = 4
    updates = max(1, updates)
    try:
        mentor_fraction = float(offline.get("mentor_fraction", 0.3))
    except Exception:
        mentor_fraction = 0.3
    mentor_fraction = float(np.clip(mentor_fraction, 0.0, 1.0))

    offline_cfg = OfflineTrainingConfig(
        enabled=offline_enabled,
        algorithm=str(offline.get("algorithm", "dqn") or "dqn"),
        train_every_episodes=train_every,
        batch_size=batch_size,
        updates=updates,
        mentor_fraction=mentor_fraction,
    )

    return StageLearningConfig(
        stage=stage,
        interactive=InteractiveLoopConfig(
            mentor_interval=mentor_interval,
            max_steps=max_steps,
            language_goal=language_goal,
        ),
        mentor=MentorConfig(
            enabled=mentor_enabled,
            reward_bonus=mentor_bonus,
            utterance=utterance,
        ),
        shaping=shaping_cfg,
        replay_capacity=replay_capacity,
        offline=offline_cfg,
    )


def build_default_reward_shaper(config: RewardShapingConfig):
    def shaper(reward: float, packet: Any, decision: Any, feedback: Dict[str, Any]) -> float:
        shaped = float(reward)
        if not config.enabled:
            return shaped

        if feedback.get("mentor"):
            shaped += float(config.mentor_bonus)
        if feedback.get("critique"):
            shaped -= float(config.critique_penalty)

        confidence = getattr(decision, "confidence", None)
        try:
            conf_value = float(confidence) if confidence is not None else 1.0
        except Exception:
            conf_value = 1.0
        if conf_value < 0.5:
            shaped -= float(config.low_confidence_penalty) * float(np.clip(0.5 - conf_value, 0.0, 0.5) / 0.5)

        rewards = getattr(packet, "rewards", None)
        if isinstance(rewards, dict) and rewards:
            numeric = [float(v) for v in rewards.values() if isinstance(v, (int, float))]
            if numeric:
                shaped += float(np.mean(numeric)) * float(config.reward_channel_weight)

        success = feedback.get("success")
        if success is None:
            success = feedback.get("task_success")
        if success is None:
            success = feedback.get("terminated_successfully")
        if isinstance(success, bool) and success:
            shaped += float(config.success_bonus)

        return float(shaped)

    return shaper


def build_default_mentor_policy(action_space: Sequence[Any], *, config: MentorConfig):
    tokens = list(action_space)

    def policy(packet: Any, decision: Any) -> TeacherSignal:
        hint = None
        if isinstance(getattr(packet, "info", None), dict):
            for key in ("teacher_action", "expert_action", "suggested_action", "action_hint"):
                if packet.info.get(key) is not None:
                    hint = packet.info.get(key)
                    break
        if hint is None and isinstance(getattr(decision, "metadata", None), dict):
            hint = decision.metadata.get("suggested_action") or decision.metadata.get("action_hint")

        action = tokens[0] if tokens else 0
        if hint is not None and tokens:
            if hint in tokens:
                action = hint
            else:
                try:
                    idx = int(hint)
                except Exception:
                    idx = None
                if idx is not None and 0 <= idx < len(tokens):
                    action = tokens[idx]

        return TeacherSignal(
            action=action,
            utterance=config.utterance,
            reward_bonus=float(config.reward_bonus),
            critique=None,
            metadata={"teacher": "heuristic"},
        )

    return policy


class ImitationOfflineTrainer:
    """Offline trainer that replays mixed mentor+policy transitions into an RL agent."""

    def __init__(self, *, policy_bridge: Any, rl_agent: Any):
        self.policy_bridge = policy_bridge
        self.rl_agent = rl_agent

    @staticmethod
    def _resolve_action_idx(transition: InteractionTransition, *, policy_bridge: Any) -> Optional[int]:
        meta = getattr(transition, "policy_metadata", None)
        if isinstance(meta, dict) and meta.get("action_idx") is not None:
            try:
                return int(meta["action_idx"])
            except Exception:
                pass

        action_space = getattr(policy_bridge, "action_space", ())
        if action_space:
            for idx, token in enumerate(action_space):
                try:
                    if token == transition.action:
                        return int(idx)
                except Exception:
                    continue

            mapper = getattr(policy_bridge, "map_action", None)
            decision = getattr(transition, "decision", None)
            if callable(mapper) and decision is not None:
                for idx, token in enumerate(action_space):
                    try:
                        if mapper(token, decision) == transition.action:
                            return int(idx)
                    except Exception:
                        continue

        return None

    @staticmethod
    def _bump_steps(agent: Any, bumps: int) -> None:
        if bumps <= 0:
            return
        if not hasattr(agent, "steps"):
            return
        interval = 1
        cfg = getattr(agent, "config", None)
        if cfg is not None and getattr(cfg, "update_interval", None) is not None:
            try:
                interval = int(getattr(cfg, "update_interval", 1))
            except Exception:
                interval = 1
        interval = max(1, interval)
        try:
            agent.steps = int(getattr(agent, "steps", 0)) + interval * int(bumps)
        except Exception:
            return

    def train(
        self,
        replay_buffer: ImitationReplayBuffer,
        *,
        batch_size: int = 32,
        updates: int = 4,
        mentor_fraction: float = 0.3,
    ) -> Dict[str, Any]:
        if replay_buffer is None or len(replay_buffer) == 0:
            return {"status": "empty_buffer", "updates": 0}

        batch_size = max(1, int(batch_size))
        updates = max(1, int(updates))
        mentor_fraction = float(np.clip(float(mentor_fraction), 0.0, 1.0))

        sampler = getattr(replay_buffer, "sample_mixed", None)
        if callable(sampler):
            batch = sampler(batch_size, mentor_fraction=mentor_fraction)
        else:
            batch = replay_buffer.sample(batch_size)

        observe = getattr(self.rl_agent, "observe", None)
        update = getattr(self.rl_agent, "update", None)
        if not callable(observe) or not callable(update):
            return {"status": "agent_not_trainable", "updates": 0, "batch": len(batch)}

        encoded = 0
        for transition in batch:
            action_idx = self._resolve_action_idx(transition, policy_bridge=self.policy_bridge)
            if action_idx is None:
                continue
            try:
                state = self.policy_bridge.encode(transition.observation, transition.decision)
                next_decision = self.policy_bridge.decision_from_packet(transition.next_observation)
                next_state = self.policy_bridge.encode(transition.next_observation, next_decision)
            except Exception:
                continue

            try:
                reward = float(getattr(transition, "shaped_reward", transition.reward))
            except Exception:
                reward = 0.0
            done = bool(getattr(transition, "terminated", False))
            try:
                observe(state, int(action_idx), reward, next_state, done)
            except Exception:
                continue
            encoded += 1

        metrics = None
        for _ in range(updates):
            self._bump_steps(self.rl_agent, 1)
            try:
                snapshot = update()
            except Exception:
                snapshot = None
            if snapshot:
                metrics = snapshot

        return {
            "status": "trained",
            "batch": len(batch),
            "encoded": encoded,
            "updates": updates,
            "metrics": metrics,
        }


class DevelopmentalLearningController:
    """Coordinate stage-aware mentor schedules and offline training for an interactive loop."""

    def __init__(
        self,
        *,
        loop: InteractiveLanguageLoop,
        stage_manager: CurriculumStageManager,
        rl_agent: Any,
        mentor_policy: Any = None,
        reward_shaper: Any = None,
    ) -> None:
        self.loop = loop
        self.stage_manager = stage_manager
        self.rl_agent = rl_agent
        self._episode_counter = 0
        self._mentor_policy_override = mentor_policy
        self._reward_shaper_override = reward_shaper
        self._offline_trainer = ImitationOfflineTrainer(
            policy_bridge=loop.policy_bridge,
            rl_agent=rl_agent,
        )

    def current_config(self) -> StageLearningConfig:
        return extract_stage_learning_config(self.stage_manager.current_config())

    def configure_loop(self, config: StageLearningConfig) -> None:
        if config.replay_capacity and getattr(self.loop.replay_buffer, "capacity", None) != config.replay_capacity:
            existing = self.loop.replay_buffer.to_list() if self.loop.replay_buffer is not None else []
            buffer = ImitationReplayBuffer(capacity=int(config.replay_capacity))
            for transition in existing[-int(config.replay_capacity) :]:
                buffer.add(transition)
            self.loop.replay_buffer = buffer

        if self._reward_shaper_override is not None:
            self.loop.reward_shaper = self._reward_shaper_override
        else:
            self.loop.reward_shaper = build_default_reward_shaper(config.shaping)

        if config.mentor.enabled and config.interactive.mentor_interval > 0:
            if self._mentor_policy_override is not None:
                self.loop.mentor_policy = self._mentor_policy_override
            else:
                action_space = getattr(self.loop.policy_bridge, "action_space", ())
                self.loop.mentor_policy = build_default_mentor_policy(
                    action_space,
                    config=config.mentor,
                )
        else:
            self.loop.mentor_policy = None

    def run_episode(self, *, language_goal: Optional[str] = None) -> Dict[str, Any]:
        config = self.current_config()
        self.configure_loop(config)

        self._episode_counter += 1
        goal = language_goal or config.interactive.language_goal

        episode = self.loop.run_episode(
            max_steps=config.interactive.max_steps,
            mentor_interval=config.interactive.mentor_interval,
            language_goal=goal,
        )

        offline_report = None
        if config.offline.enabled and self._episode_counter % config.offline.train_every_episodes == 0:
            offline_report = self._offline_trainer.train(
                self.loop.replay_buffer,
                batch_size=config.offline.batch_size,
                updates=config.offline.updates,
                mentor_fraction=config.offline.mentor_fraction,
            )

        return {
            "stage": config.stage,
            "episode": episode,
            "offline": offline_report,
        }


__all__ = [
    "DevelopmentalLearningController",
    "ImitationOfflineTrainer",
    "StageLearningConfig",
    "extract_stage_learning_config",
]

