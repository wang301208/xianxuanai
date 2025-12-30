"""Adapters bridging AutoGPT agents and the WholeBrain simulation stack."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, List

from modules.interface import ModuleInterface
try:  # optional dependency
    from backend.capability.skill_registry import get_skill_registry
except Exception:  # pragma: no cover - registry optional
    get_skill_registry = None  # type: ignore

from .state import BrainCycleResult

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from third_party.autogpt.autogpt.agents.base import BaseAgent


logger = logging.getLogger(__name__)


class BrainModule(ModuleInterface):
    """Thin adapter exposing brain functionality via ModuleInterface."""

    # For demonstration purposes the brain depends on the evolution module.
    dependencies = ["evolution"]

    def __init__(self) -> None:
        self.initialized = False

    def initialize(self) -> None:  # pragma: no cover - trivial
        self.initialized = True

    def shutdown(self) -> None:  # pragma: no cover - trivial
        self.initialized = False


def _clamp_unit(value: float | int, scale: float = 1.0) -> float:
    """Clamp ``value`` into the [0, 1] range using ``scale`` as denominator."""

    if scale == 0:
        return 0.0
    try:
        return max(0.0, min(1.0, float(value) / float(scale)))
    except (TypeError, ValueError):  # pragma: no cover - defensive programming
        return 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


class WholeBrainAgentAdapter:
    """Prepare observations for :class:`WholeBrainSimulation` and decode responses."""

    def __init__(self, agent: "BaseAgent") -> None:
        self._agent = agent

    # ------------------------------------------------------------------
    # Input construction helpers
    # ------------------------------------------------------------------
    def build_cycle_input(self, instruction: str | None = None) -> Dict[str, Any]:
        """Translate the agent's latest context into WholeBrain sensory inputs."""

        agent = self._agent
        config = agent.config
        episodes = list(agent.event_history.episodes)
        total = len(episodes)
        successes = sum(
            1 for ep in episodes if getattr(getattr(ep, "result", None), "status", "") == "success"
        )
        failures = sum(
            1 for ep in episodes if getattr(getattr(ep, "result", None), "status", "") == "error"
        )
        interrupts = sum(
            1
            for ep in episodes
            if getattr(getattr(ep, "result", None), "status", "") == "interrupted_by_human"
        )
        recent = episodes[-3:]
        last_episode = episodes[-1] if episodes else None
        success_ratio = _safe_ratio(successes, total)
        failure_ratio = _safe_ratio(failures, total)

        goals = list(getattr(agent.ai_profile, "ai_goals", []) or [])
        directives = list(getattr(agent.directives, "general_guidelines", []) or [])

        # Numerical proxies for sensory channels
        vision = [
            _clamp_unit(config.cycle_count, 25.0),
            _clamp_unit(total, 25.0),
            _clamp_unit(success_ratio, 1.0),
            _clamp_unit(failure_ratio, 1.0),
        ]
        unique_actions = len(
            {
                getattr(ep.action, "name", "unknown")
                for ep in episodes
                if getattr(ep, "action", None)
            }
        )
        auditory = [
            _clamp_unit(len(goals), 10.0),
            _clamp_unit(len(directives), 10.0),
            _clamp_unit(unique_actions, 25.0),
        ]
        budget = config.cycle_budget if config.cycle_budget is not None else 0
        somatosensory = [
            _clamp_unit(budget, 50.0),
            _clamp_unit(interrupts, max(1, total)),
            _clamp_unit(success_ratio - failure_ratio + 0.5, 1.0),
        ]

        task = getattr(agent.state, "task", None)
        task_input = getattr(task, "input", None)
        additional_input = getattr(task, "additional_input", None)

        history_lines: List[str] = []
        for episode in recent:
            action = getattr(episode, "action", None)
            name = getattr(action, "name", "unknown")
            status = getattr(getattr(episode, "result", None), "status", "pending")
            reasoning = getattr(action, "reasoning", "")
            summary = episode.summary or reasoning or status
            history_lines.append(f"{name}:{status} -> {summary}")

        text_chunks: List[str] = []
        if instruction:
            text_chunks.append(str(instruction))
        if task_input:
            text_chunks.append(f"Task: {task_input}")
        if additional_input:
            text_chunks.append(f"Additional context: {additional_input}")
        if goals:
            text_chunks.append("Goals: " + "; ".join(goals[:3]))
        if history_lines:
            text_chunks.append("Recent episodes: " + " | ".join(history_lines))
        if last_episode and last_episode.result and getattr(last_episode.result, "status", ""):
            text_chunks.append(f"Last outcome: {last_episode.result}")

        text_payload = "\n".join(chunk for chunk in text_chunks if chunk)
        novelty_proxy = _safe_ratio(unique_actions, max(1, total))

        context: Dict[str, float] = {
            "success_rate": success_ratio,
            "failure_rate": failure_ratio,
            "progress": success_ratio - failure_ratio,
            "cycle_count": float(config.cycle_count),
            "novelty": novelty_proxy,
            "threat": failure_ratio,
            "safety": max(0.0, 1.0 - failure_ratio),
        }
        if last_episode and last_episode.action and last_episode.action.reasoning:
            context["reasoning_length"] = _clamp_unit(len(last_episode.action.reasoning), 500.0)

        input_payload: Dict[str, Any] = {
            "agent_id": agent.state.agent_id,
            "text": text_payload,
            "context": context,
            "vision": vision,
            "auditory": auditory,
            "somatosensory": somatosensory,
            "is_salient": failure_ratio > 0.2 or interrupts > 0,
        }
        if task_input:
            input_payload["task"] = task_input
        if goals:
            input_payload.setdefault("context", {})["goal_focus"] = goals[0]
        if additional_input:
            input_payload.setdefault("context", {})["user_hint"] = 1.0

        if get_skill_registry is not None:
            try:
                registry = get_skill_registry()
                available_skills = [
                    {
                        "name": spec.name,
                        "description": spec.description,
                        "provider": spec.provider,
                        "version": spec.version,
                        "cost": spec.cost,
                        "tags": spec.tags,
                    }
                    for spec in registry.list_specs()
                    if spec.enabled
                ]
            except Exception:  # pragma: no cover - registry lookup best effort
                available_skills = []
            if available_skills:
                input_payload.setdefault("context", {})["available_skills"] = available_skills[:8]

        return input_payload

    # ------------------------------------------------------------------
    # Output decoding helpers
    # ------------------------------------------------------------------
    def translate_cycle_result(
        self, result: BrainCycleResult
    ) -> tuple[str, Dict[str, str], Dict[str, Any]]:
        """Map :class:`BrainCycleResult` into the tuple expected by the agent loop."""

        command_name = "internal_brain_action"
        plan_list = list(result.intent.plan)
        plan_summary = result.metadata.get("cognitive_plan") or ", ".join(plan_list)

        command_args: Dict[str, str] = {
            "intention": result.intent.intention,
            "confidence": f"{float(result.intent.confidence):.2f}",
        }
        if plan_summary:
            command_args["plan"] = plan_summary

        emotion_primary = (
            result.emotion.primary.value
            if hasattr(result.emotion.primary, "value")
            else str(result.emotion.primary)
        )
        emotion_payload = {
            "primary": emotion_primary,
            "intensity": float(result.emotion.intensity),
            "mood": float(result.emotion.mood),
            "dimensions": {k: float(v) for k, v in result.emotion.dimensions.items()},
            "context": {k: float(v) for k, v in result.emotion.context.items()},
            "decay": float(result.emotion.decay),
        }

        curiosity_payload = {
            "drive": float(result.curiosity.drive),
            "fatigue": float(result.curiosity.fatigue),
            "novelty": float(result.curiosity.last_novelty),
        }

        personality_payload = asdict(result.personality)

        thoughts: Dict[str, Any] = {
            "backend": "whole_brain",
            "text": (result.thoughts.summary if result.thoughts and result.thoughts.summary else plan_summary)
            or result.intent.intention,
            "reasoning": "; ".join(result.thoughts.plan) if result.thoughts else "; ".join(plan_list),
            "plan": plan_list,
            "confidence": float(result.intent.confidence),
            "weights": {k: float(v) for k, v in result.intent.weights.items()},
            "tags": list(result.intent.tags),
            "emotion": emotion_payload,
            "curiosity": curiosity_payload,
            "personality": personality_payload,
            "metrics": {k: float(v) for k, v in result.metrics.items()},
            "metadata": {k: v for k, v in result.metadata.items() if v is not None},
        }

        if result.thoughts:
            thoughts["focus"] = result.thoughts.focus
            thoughts["thought_plan"] = list(result.thoughts.plan)
            thoughts["memory_refs"] = list(result.thoughts.memory_refs)
            thoughts["thought_tags"] = list(result.thoughts.tags)
        if result.feeling:
            thoughts["feeling"] = {
                "descriptor": result.feeling.descriptor,
                "valence": float(result.feeling.valence),
                "arousal": float(result.feeling.arousal),
                "mood": float(result.feeling.mood),
                "confidence": float(result.feeling.confidence),
                "context_tags": list(result.feeling.context_tags),
            }

        thoughts["speech"] = (
            result.feeling.descriptor
            if result.feeling
            else f"Act with intention {result.intent.intention}"
        )
        thoughts["energy_used"] = int(result.energy_used)
        thoughts["idle_skipped"] = int(result.idle_skipped)

        perception_snapshot = getattr(result, "perception", None)
        if perception_snapshot is not None:
            knowledge_facts = getattr(perception_snapshot, "knowledge_facts", []) or []
            if knowledge_facts:
                try:
                    self._agent._pending_knowledge_facts.extend(knowledge_facts)
                except AttributeError:
                    logger.debug("Agent missing pending knowledge queue.")
            semantic_map = getattr(perception_snapshot, "semantic", {}) or {}
            if semantic_map:
                summaries = [
                    str(value.get("summary")).strip()
                    for value in semantic_map.values()
                    if value.get("summary")
                ]
                if summaries:
                    try:
                        self._agent._pending_knowledge_statements.extend(summaries)
                    except AttributeError:
                        logger.debug("Agent missing pending knowledge statements store.")

        return command_name, command_args, thoughts


__all__ = ["BrainModule", "WholeBrainAgentAdapter"]
