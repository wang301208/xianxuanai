"""
Goal-driven attention controller.

This module produces lightweight directives that bias perceptual and semantic
attention towards goal-relevant concepts, modalities, and workspace items.  It
is intentionally heuristic so it can operate without additional training data
while still simulating top-down control.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence


def _normalise_goals(goals: Any) -> List[str]:
    if goals is None:
        return []
    if isinstance(goals, str):
        return [goals.strip()] if goals.strip() else []
    if isinstance(goals, Sequence):
        items: List[str] = []
        for entry in goals:
            if isinstance(entry, str):
                value = entry.strip()
            elif isinstance(entry, dict):
                value = (
                    entry.get("name")
                    or entry.get("goal")
                    or entry.get("description")
                    or ""
                )
                value = str(value).strip()
            else:
                value = str(entry).strip()
            if value:
                items.append(value)
        return items
    return [str(goals)]


class GoalDrivenAttentionController:
    """Compute top-down attention directives given task context."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.default_modality_weights: Dict[str, float] = {
            "vision": float(cfg.get("vision_weight", 0.6)),
            "auditory": float(cfg.get("auditory_weight", 0.4)),
            "language": float(cfg.get("language_weight", 0.7)),
            "structured": float(cfg.get("structured_weight", 0.55)),
        }
        self.max_focus_terms = int(cfg.get("max_focus_terms", 6))
        self.workspace_focus_cap = int(cfg.get("workspace_focus_cap", 4))
        self.goal_keyword_boost = float(cfg.get("goal_keyword_boost", 0.15))
        self.topic_weight = float(cfg.get("topic_weight", 0.25))

    # ------------------------------------------------------------------ #
    def compute(
        self,
        *,
        goals: Any = None,
        planner: Optional[Dict[str, Any]] = None,
        dialogue_state: Optional[Dict[str, Any]] = None,
        working_memory: Optional[Dict[str, Any]] = None,
        motivation: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        directives: Dict[str, Any] = {}

        normalised_goals = _normalise_goals(goals)
        focus_terms = self._derive_focus_terms(normalised_goals, planner, dialogue_state, working_memory)
        modality_weights = self._derive_modality_weights(normalised_goals, dialogue_state)
        workspace_attention = self._build_workspace_attention(normalised_goals, motivation)

        directives["semantic_focus"] = focus_terms
        directives["modality_weights"] = modality_weights
        directives["workspace_attention"] = workspace_attention
        directives["workspace_focus"] = focus_terms[: self.workspace_focus_cap]
        directives["goal_snapshot"] = normalised_goals[:4]
        return directives

    # ------------------------------------------------------------------ #
    def _derive_focus_terms(
        self,
        goals: Sequence[str],
        planner: Optional[Dict[str, Any]],
        dialogue_state: Optional[Dict[str, Any]],
        working_memory: Optional[Dict[str, Any]],
    ) -> List[str]:
        focus_terms: List[str] = []

        def extend_terms(items: Iterable[str], weight: float = 1.0) -> None:
            for item in items:
                token = str(item).strip()
                if not token:
                    continue
                if token.lower() not in (term.lower() for term in focus_terms):
                    focus_terms.append(token if weight >= 1.0 else f"{token}")

        for goal in goals:
            tokens = [token for token in goal.replace("_", " ").split() if len(token) > 2]
            extend_terms(tokens)

        if planner:
            candidates = planner.get("candidates") or []
            for candidate in candidates:
                action = candidate.get("action") if isinstance(candidate, dict) else candidate
                if action:
                    extend_terms(
                        [action] + candidate.get("justification", []) if isinstance(candidate, dict) else [action]
                    )

        if dialogue_state:
            extend_terms(dialogue_state.get("topics", [])[: self.max_focus_terms], self.topic_weight)
            extend_terms(dialogue_state.get("entities", [])[: self.max_focus_terms])

        if working_memory:
            extend_terms(working_memory.get("key_terms", [])[: self.max_focus_terms])
            extend_terms(working_memory.get("pending_actions", [])[: self.max_focus_terms])

        return focus_terms[: self.max_focus_terms]

    def _derive_modality_weights(
        self,
        goals: Sequence[str],
        dialogue_state: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        weights = dict(self.default_modality_weights)
        goal_text = " ".join(goals).lower()
        if "visual" in goal_text or "see" in goal_text:
            weights["vision"] = min(1.0, weights.get("vision", 0.6) + self.goal_keyword_boost)
        if "listen" in goal_text or "audio" in goal_text:
            weights["auditory"] = min(1.0, weights.get("auditory", 0.4) + self.goal_keyword_boost)
        if "analyz" in goal_text or "report" in goal_text or "question" in goal_text:
            weights["language"] = min(1.0, weights.get("language", 0.7) + self.goal_keyword_boost)
        if "database" in goal_text or "knowledge" in goal_text:
            weights["structured"] = min(1.0, weights.get("structured", 0.55) + self.goal_keyword_boost)

        if dialogue_state and dialogue_state.get("last_intent") == "question":
            weights["language"] = min(1.0, weights.get("language", 0.7) + 0.1)

        return weights

    def _build_workspace_attention(
        self,
        goals: Sequence[str],
        motivation: Optional[Dict[str, float]],
    ) -> Dict[str, Any]:
        if goals:
            primary = goals[0]
        else:
            primary = None
        motive = float(motivation.get(primary, 0.5)) if motivation and primary else 0.5
        return {
            "goal": primary,
            "priority": float(min(1.0, 0.5 + motive)),
        }


__all__ = ["GoalDrivenAttentionController"]

