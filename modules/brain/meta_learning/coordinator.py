"""Meta-learning coordinator enabling cross-task adaptation via RL."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence

try:  # Optional skill registry integration
    from backend.capability.skill_registry import get_skill_registry
except Exception:  # pragma: no cover - optional dependency
    get_skill_registry = None  # type: ignore[assignment]

try:  # pragma: no cover - skill registry may not be available during tests
    from modules.skills import SkillRegistry, SkillSpec
except Exception:  # pragma: no cover - degrade gracefully without skill support
    SkillRegistry = None  # type: ignore[assignment]
    SkillSpec = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _normalise_token(token: str) -> str:
    token = (token or "").strip().lower()
    normalised = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "-" for ch in token)
    return normalised.strip("-_") or "skill"


@dataclass
class TaskExperience:
    """Record of a self-directed task execution."""

    state_signature: str
    intention: str
    plan: List[str] = field(default_factory=list)
    reward: float = 0.0
    success: bool = False
    context: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    cycle_index: int = 0
    timestamp: float = field(default_factory=time.time)
    origin: str = "meta-learning"

    def summary(self) -> Dict[str, Any]:
        return {
            "state": self.state_signature,
            "intention": self.intention,
            "plan": list(self.plan),
            "reward": float(self.reward),
            "success": bool(self.success),
            "cycle_index": int(self.cycle_index),
            "timestamp": float(self.timestamp),
        }


class MetaLearningCoordinator:
    """Bridge between WholeBrain cycles, RL policy updates, and skills."""

    def __init__(
        self,
        *,
        buffer_size: int = 128,
        success_threshold: float = 0.6,
        min_successes_to_register: int = 2,
        policy: Any | None = None,
        skill_registry: Any | None = None,
    ) -> None:
        self._experiences: Deque[TaskExperience] = deque(maxlen=max(1, int(buffer_size)))
        self._policy = policy
        self._skill_registry = skill_registry or self._resolve_skill_registry()
        self._success_threshold = max(0.0, min(1.0, float(success_threshold)))
        self._min_successes = max(1, int(min_successes_to_register))
        self.learned_skills: Dict[str, Dict[str, Any]] = {}
        self.last_update: Dict[str, Any] | None = None

    # ------------------------------------------------------------------
    def bind_policy(self, policy: Any) -> None:
        """Attach the active cognitive policy for reinforcement updates."""

        self._policy = policy

    # ------------------------------------------------------------------
    def attach_skill_registry(self, registry: Any) -> None:
        """Override the skill registry used for persistence."""

        self._skill_registry = registry

    # ------------------------------------------------------------------
    def inject_suggestions(
        self,
        context: Dict[str, Any],
        perception: Any,
        history: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Augment context with high-confidence learned skills."""

        if not self.learned_skills:
            return []
        ranked = sorted(
            self.learned_skills.items(),
            key=lambda item: item[1].get("success_rate", 0.0),
            reverse=True,
        )
        suggestions: List[Dict[str, Any]] = []
        for key, data in ranked[:3]:
            exp: TaskExperience | None = data.get("experience")
            if exp is None:
                continue
            suggestions.append(
                {
                    "skill_key": key,
                    "intention": exp.intention,
                    "plan": list(exp.plan),
                    "success_rate": float(data.get("success_rate", 0.0)),
                    "skill_id": data.get("skill_id"),
                }
            )
        if suggestions:
            context.setdefault("meta_skill_suggestions", suggestions)
        return suggestions

    # ------------------------------------------------------------------
    def record_outcome(
        self,
        *,
        cycle_index: int,
        state_signature: str,
        decision: Mapping[str, Any],
        feedback_metrics: Optional[Mapping[str, Any]],
        reward_signal: Optional[float],
        cognitive_context: Mapping[str, Any] | None = None,
    ) -> Optional[Dict[str, Any]]:
        """Persist outcome information and train the reinforcement policy."""

        intention = str(decision.get("intention", "") or "")
        if not intention:
            return None
        plan = list(decision.get("plan") or [])
        context = dict(cognitive_context or {})
        metrics = dict(feedback_metrics or {})
        reward = self._extract_reward(metrics, reward_signal)
        success = self._extract_success(metrics, reward)
        experience = TaskExperience(
            state_signature=state_signature or context.get("task", intention) or intention,
            intention=intention,
            plan=plan,
            reward=reward,
            success=success,
            context=context,
            metrics=metrics,
            cycle_index=int(cycle_index),
        )
        self._experiences.append(experience)
        self._update_policy(reward, success, experience)
        skill_entry = self._update_skill_statistics(experience)
        update_payload = experience.summary()
        update_payload["success_rate"] = float(skill_entry.get("success_rate", 0.0))
        update_payload["skill_id"] = skill_entry.get("skill_id")
        self.last_update = dict(update_payload)
        return self.last_update

    # ------------------------------------------------------------------
    def _update_policy(self, reward: float, success: bool, experience: TaskExperience) -> None:
        policy = self._policy
        if policy is None:
            return
        if hasattr(policy, "integrate_external_feedback"):
            try:
                policy.integrate_external_feedback(
                    reward,
                    success,
                    metadata={
                        "state": experience.state_signature,
                        "plan": list(experience.plan),
                        "cycle_index": experience.cycle_index,
                    },
                )
            except Exception:  # pragma: no cover - defensive logging
                logger.debug("Failed to integrate external feedback into policy.", exc_info=True)
        if hasattr(policy, "replay_from_buffer"):
            try:
                policy.replay_from_buffer()
            except Exception:  # pragma: no cover - defensive logging
                logger.debug("Policy replay_from_buffer failed.", exc_info=True)

    # ------------------------------------------------------------------
    def _update_skill_statistics(self, experience: TaskExperience) -> Dict[str, Any]:
        key = self._skill_key(experience)
        entry = self.learned_skills.get(key)
        if entry is None:
            entry = {
                "experience": experience,
                "successes": 0,
                "failures": 0,
                "uses": 0,
                "skill_id": None,
                "success_rate": 0.0,
            }
        entry["experience"] = experience
        entry["uses"] = int(entry.get("uses", 0)) + 1
        if experience.success:
            entry["successes"] = int(entry.get("successes", 0)) + 1
            entry["last_success_cycle"] = experience.cycle_index
        else:
            entry["failures"] = int(entry.get("failures", 0)) + 1
            entry["last_failure_cycle"] = experience.cycle_index
        total = max(1, entry["uses"])
        entry["success_rate"] = entry.get("successes", 0) / total
        if (
            experience.success
            and entry.get("skill_id") is None
            and entry["successes"] >= self._min_successes
            and entry["success_rate"] >= self._success_threshold
            and experience.plan
        ):
            skill_id = self._register_skill(key, experience)
            entry["skill_id"] = skill_id
        self.learned_skills[key] = entry
        return entry

    # ------------------------------------------------------------------
    def _register_skill(self, key: str, experience: TaskExperience) -> Optional[str]:
        registry = self._skill_registry
        if registry is None or SkillSpec is None:
            return None
        skill_name = f"meta::{key}"
        description = (
            f"Auto-generated plan for intention '{experience.intention}' with state "
            f"'{experience.state_signature}'."
        )
        spec = SkillSpec(
            name=skill_name,
            description=description,
            execution_mode="local",
            input_schema={
                "type": "object",
                "properties": {
                    "context": {"type": "object"},
                },
            },
            output_schema={
                "type": "object",
                "properties": {
                    "intention": {"type": "string"},
                    "plan": {"type": "array", "items": {"type": "string"}},
                },
            },
            tags=["meta-learning", experience.intention],
            provider="meta-learning",
            version="0.1.0",
        )

        def _handler(payload: Dict[str, Any], *, plan: List[str] | None = None) -> Dict[str, Any]:
            return {
                "intention": experience.intention,
                "plan": list(plan or experience.plan),
                "context": dict(payload.get("context", {})),
            }

        try:
            registry.register(
                spec,
                handler=_handler,
                replace=True,
                metadata={
                    "source": "meta-learning",
                    "state_signature": experience.state_signature,
                    "success_rate": experience.success,
                },
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Failed to register learned skill '%s'.", skill_name, exc_info=True)
            return None
        return skill_name

    # ------------------------------------------------------------------
    def _extract_reward(
        self,
        feedback_metrics: Mapping[str, Any],
        reward_signal: Optional[float],
    ) -> float:
        if "reward" in feedback_metrics:
            try:
                return float(feedback_metrics["reward"])
            except (TypeError, ValueError):
                return float(reward_signal or 0.0)
        if "success" in feedback_metrics:
            success = bool(feedback_metrics["success"])
            base = 1.0 if success else -0.5
            return base if reward_signal is None else float(reward_signal)
        return float(reward_signal or 0.0)

    # ------------------------------------------------------------------
    def _extract_success(self, metrics: Mapping[str, Any], reward: float) -> bool:
        if "success" in metrics:
            try:
                return bool(metrics["success"])
            except Exception:  # pragma: no cover - defensive conversion
                return reward > 0
        return reward > 0

    # ------------------------------------------------------------------
    def _skill_key(self, experience: TaskExperience) -> str:
        anchor = experience.state_signature or experience.plan[0] if experience.plan else experience.intention
        return f"{experience.intention}:{_normalise_token(str(anchor))}"

    # ------------------------------------------------------------------
    def _resolve_skill_registry(self) -> Any:
        if get_skill_registry is None:
            return None
        try:
            return get_skill_registry()
        except Exception:  # pragma: no cover - optional dependency path
            logger.debug("Meta-learning skill registry unavailable.", exc_info=True)
            return None


__all__ = ["MetaLearningCoordinator", "TaskExperience"]

