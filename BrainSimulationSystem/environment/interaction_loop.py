"""Interactive environment loop for language-grounded reinforcement and imitation.

This module coordinates closed-loop interaction between the environment,
reinforcement learning policies, and optional mentor/teacher signals. It is
intended to support the "trial-and-error + imitation" developmental path where
an agent describes its intent, executes an action, receives feedback, and uses
both rewards and demonstrations to refine behaviour.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from BrainSimulationSystem.environment.base import EnvironmentController, PerceptionPacket
from BrainSimulationSystem.environment.policy_bridge import HierarchicalPolicyBridge, HighLevelDecision
from BrainSimulationSystem.learning.feedback_store import FeedbackLogger

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from BrainSimulationSystem.models.language_hub import LanguageHub

ActionSelector = Callable[[np.ndarray], Tuple[int, Dict[str, float]]]
TeacherPolicy = Callable[[PerceptionPacket, HighLevelDecision], "TeacherSignal"]
RewardShaper = Callable[[float, PerceptionPacket, HighLevelDecision, Dict[str, Any]], float]


@dataclass
class TeacherSignal:
    """Feedback returned by a mentor/teacher policy.

    Attributes:
        action: Low-level environment action to execute.
        utterance: Optional language demonstration or correction.
        reward_bonus: Additional reward signal provided by the mentor.
        critique: Free-form feedback text.
        metadata: Extra fields forwarded to transition logs.
    """

    action: Any
    utterance: Optional[str] = None
    reward_bonus: float = 0.0
    critique: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InteractionTransition:
    """Single transition captured during interactive learning."""

    observation: PerceptionPacket
    next_observation: PerceptionPacket
    decision: HighLevelDecision
    action: Any
    action_source: str
    utterance: Optional[str]
    reward: float
    shaped_reward: float
    terminated: bool
    feedback: Dict[str, Any] = field(default_factory=dict)
    policy_metadata: Dict[str, Any] = field(default_factory=dict)


class ImitationReplayBuffer:
    """Simple replay buffer mixing mentor demonstrations and agent rollouts."""

    def __init__(self, capacity: int = 2048) -> None:
        self.capacity = int(capacity)
        self._buffer: Deque[InteractionTransition] = deque(maxlen=self.capacity)

    def add(self, transition: InteractionTransition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int) -> List[InteractionTransition]:
        batch_size = int(batch_size)
        if batch_size <= 0 or len(self._buffer) == 0:
            return []
        indices = np.random.choice(len(self._buffer), size=min(batch_size, len(self._buffer)), replace=False)
        return [self._buffer[idx] for idx in indices]

    def sample_mixed(self, batch_size: int, *, mentor_fraction: float = 0.5) -> List[InteractionTransition]:
        """Sample a batch with a target mentor-vs-policy mix."""

        batch_size = int(batch_size)
        if batch_size <= 0 or len(self._buffer) == 0:
            return []

        try:
            frac = float(mentor_fraction)
        except Exception:
            frac = 0.5
        frac = float(np.clip(frac, 0.0, 1.0))

        mentor_pool = [t for t in self._buffer if getattr(t, "action_source", "") == "mentor"]
        policy_pool = [t for t in self._buffer if getattr(t, "action_source", "") != "mentor"]

        desired_mentor = int(round(batch_size * frac))
        desired_policy = batch_size - desired_mentor

        mentor_count = min(desired_mentor, len(mentor_pool))
        policy_count = min(desired_policy, len(policy_pool))

        remaining = batch_size - (mentor_count + policy_count)
        if remaining > 0:
            if len(mentor_pool) - mentor_count >= len(policy_pool) - policy_count:
                mentor_count = min(len(mentor_pool), mentor_count + remaining)
            else:
                policy_count = min(len(policy_pool), policy_count + remaining)

        samples: List[InteractionTransition] = []
        if mentor_count > 0:
            indices = np.random.choice(len(mentor_pool), size=mentor_count, replace=False)
            samples.extend([mentor_pool[idx] for idx in indices])
        if policy_count > 0:
            indices = np.random.choice(len(policy_pool), size=policy_count, replace=False)
            samples.extend([policy_pool[idx] for idx in indices])

        if len(samples) > 1:
            np.random.shuffle(samples)
        return samples

    def to_list(self) -> List[InteractionTransition]:
        return list(self._buffer)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._buffer)


class InteractiveLanguageLoop:
    """
    Closed-loop interaction coordinator for language-conditioned RL/imitation.

    This helper handles environment reset/step calls, queries an RL policy via a
    :class:`HierarchicalPolicyBridge`, optionally requests mentor demonstrations,
    generates task-oriented utterances via :class:`LanguageHub`, and records
    transitions for both reinforcement and imitation learning.
    """

    def __init__(
        self,
        controller: EnvironmentController,
        policy_bridge: HierarchicalPolicyBridge,
        *,
        rl_selector: ActionSelector,
        language_hub: "LanguageHub" | None = None,
        mentor_policy: TeacherPolicy | None = None,
        reward_shaper: RewardShaper | None = None,
        replay_buffer: ImitationReplayBuffer | None = None,
        feedback_logger: FeedbackLogger | None = None,
    ) -> None:
        self.controller = controller
        self.policy_bridge = policy_bridge
        self.rl_selector = rl_selector
        self.language_hub = language_hub
        self.mentor_policy = mentor_policy
        self.reward_shaper = reward_shaper or self._default_reward_shaper
        self.replay_buffer = replay_buffer or ImitationReplayBuffer()
        self.feedback_logger = feedback_logger

    # ------------------------------------------------------------------ #
    def run_episode(
        self,
        *,
        max_steps: int = 32,
        mentor_interval: int = 0,
        language_goal: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute one interactive episode with optional mentor involvement."""

        packet = self.controller.reset()
        episode_log: List[InteractionTransition] = []
        total_reward = 0.0
        mentor_interval = max(0, int(mentor_interval))

        for step_idx in range(max_steps):
            use_mentor = self.mentor_policy is not None and mentor_interval > 0 and step_idx % mentor_interval == 0
            transition = self._run_step(packet, use_mentor=use_mentor, language_goal=language_goal)
            episode_log.append(transition)
            total_reward += transition.shaped_reward
            self.replay_buffer.add(transition)
            if self.feedback_logger is not None:
                self.feedback_logger.record(
                    {
                        "utterance": transition.utterance,
                        "reward": transition.shaped_reward,
                        "action_source": transition.action_source,
                        "terminated": transition.terminated,
                        "decision_intent": transition.decision.intent,
                        "feedback": transition.feedback,
                    }
                )
            if transition.terminated:
                break
            packet = transition.next_observation

        return {
            "total_reward": total_reward,
            "steps": len(episode_log),
            "transitions": episode_log,
        }

    # ------------------------------------------------------------------ #
    def _run_step(
        self,
        packet: PerceptionPacket,
        *,
        use_mentor: bool = False,
        language_goal: Optional[str] = None,
    ) -> InteractionTransition:
        decision = self.policy_bridge.decision_from_packet(packet)

        mentor_feedback: Dict[str, Any] = {}
        utterance: Optional[str] = None
        policy_metadata: Dict[str, Any] = {}

        if use_mentor and self.mentor_policy is not None:
            teacher_signal = self.mentor_policy(packet, decision)
            action = teacher_signal.action
            utterance = teacher_signal.utterance
            action_idx = None
            try:
                action_space = getattr(self.policy_bridge, "action_space", ())
                for idx, token in enumerate(action_space):
                    try:
                        if token == action:
                            action_idx = idx
                            break
                    except Exception:
                        continue
                if action_idx is None and action_space:
                    mapper = getattr(self.policy_bridge, "map_action", None)
                    if callable(mapper):
                        for idx, token in enumerate(action_space):
                            try:
                                if mapper(token, decision) == action:
                                    action_idx = idx
                                    break
                            except Exception:
                                continue
            except Exception:
                action_idx = None
            mentor_feedback = {
                "mentor": True,
                "critique": teacher_signal.critique,
                **(teacher_signal.metadata or {}),
            }
            if action_idx is not None:
                policy_metadata["action_idx"] = int(action_idx)
            policy_metadata["mentor"] = True
            base_reward_bonus = float(teacher_signal.reward_bonus)
        else:
            action, policy_metadata = self._select_action(packet, decision)
            base_reward_bonus = 0.0

        next_packet, env_reward, terminated, info = self.controller.step(action)
        reward = env_reward + base_reward_bonus

        if utterance is None and self.language_hub is not None:
            utterance = self._generate_utterance(language_goal or decision.intent, packet, decision)

        feedback = self._merge_feedback(mentor_feedback, info, packet)
        shaped_reward = self.reward_shaper(reward, packet, decision, feedback)

        transition = InteractionTransition(
            observation=packet,
            next_observation=next_packet,
            decision=decision,
            action=action,
            action_source="mentor" if use_mentor else "policy",
            utterance=utterance,
            reward=reward,
            shaped_reward=shaped_reward,
            terminated=terminated,
            feedback=feedback,
            policy_metadata=policy_metadata,
        )
        return transition

    def _select_action(self, packet: PerceptionPacket, decision: HighLevelDecision) -> Tuple[Any, Dict[str, Any]]:
        action, action_idx, obs_vector, _, metadata = self.policy_bridge.select_action(packet, _RLSelectorWrapper(self.rl_selector))
        policy_metadata: Dict[str, Any] = dict(metadata or {}) if isinstance(metadata, dict) else {}
        policy_metadata.setdefault("action_idx", int(action_idx))
        if hasattr(self.rl_selector, "record_outcome"):
            try:
                predicted = metadata.get("predicted_reward", 0.0) if isinstance(metadata, dict) else 0.0
                self.rl_selector.record_outcome(obs_vector, float(predicted))
            except Exception:
                pass
        return action, policy_metadata

    def _generate_utterance(
        self,
        goal: str,
        packet: PerceptionPacket,
        decision: HighLevelDecision,
    ) -> Optional[str]:
        context = {"environment": packet.info, "state_vector": packet.state_vector}
        try:
            generation = self.language_hub.production.generate(
                {"intent": goal, "metadata": decision.metadata},
                {"language_context": context},
                {"intent": decision.intent, "confidence": decision.confidence},
            )
            return generation.get("reply")
        except Exception:
            return None

    @staticmethod
    def _merge_feedback(mentor_feedback: Dict[str, Any], info: Dict[str, Any], packet: PerceptionPacket) -> Dict[str, Any]:
        feedback: Dict[str, Any] = {}
        feedback.update(mentor_feedback)
        feedback.update(info or {})
        if packet.rewards:
            feedback.setdefault("reward_channels", dict(packet.rewards))
        return feedback

    @staticmethod
    def _default_reward_shaper(
        reward: float,
        packet: PerceptionPacket,
        decision: HighLevelDecision,
        feedback: Dict[str, Any],
    ) -> float:
        shaped = reward
        if feedback.get("mentor"):
            shaped += 0.05
        if feedback.get("critique"):
            shaped -= 0.05
        if decision.confidence < 0.5:
            shaped -= 0.02
        if packet.rewards:
            shaped += float(np.mean(list(packet.rewards.values()))) * 0.1
        return float(shaped)


class _RLSelectorWrapper:
    """Adapter to provide ``select_action`` for :class:`HierarchicalPolicyBridge`."""

    def __init__(self, selector: ActionSelector) -> None:
        self._selector = selector

    def select_action(self, obs_vector: np.ndarray) -> Tuple[int, Dict[str, float]]:
        return self._selector(obs_vector)

    def record_outcome(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - passthrough
        if hasattr(self._selector, "record_outcome"):
            self._selector.record_outcome(*args, **kwargs)


__all__ = [
    "InteractiveLanguageLoop",
    "ImitationReplayBuffer",
    "InteractionTransition",
    "TeacherSignal",
]
