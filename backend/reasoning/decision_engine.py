from __future__ import annotations

"""Utilities for evaluating reasoning paths and selecting actions."""

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, Optional, Tuple


@dataclass
class ActionPlan:
    """Represents a candidate action to evaluate."""

    action: str
    utility: float
    cost: float
    rationale: str = ""


@dataclass
class ActionDirective:
    """Outcome of a high-level decision review for a command."""

    approved: bool
    command_name: Optional[str] = None
    command_args: Optional[Dict[str, str]] = None
    rationale: Optional[str] = None
    requires_replan: bool = False
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def approve(
        cls,
        command_name: str,
        command_args: Dict[str, str],
        *,
        rationale: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ActionDirective":
        return cls(
            approved=True,
            command_name=command_name,
            command_args=dict(command_args),
            rationale=rationale,
            requires_replan=False,
            metadata=dict(metadata) if metadata else None,
        )

    @classmethod
    def replan(
        cls, rationale: str, *, metadata: Optional[Dict[str, Any]] = None
    ) -> "ActionDirective":
        return cls(
            approved=False,
            command_name=None,
            command_args=None,
            rationale=rationale,
            requires_replan=True,
            metadata=dict(metadata) if metadata else None,
        )

    def resolve(self, default_name: str, default_args: Dict[str, str]) -> "ActionDirective":
        """Return a copy of the directive with missing command data filled in."""

        return ActionDirective(
            approved=self.approved,
            command_name=self.command_name or default_name,
            command_args=dict(self.command_args)
            if self.command_args is not None
            else dict(default_args),
            rationale=self.rationale,
            requires_replan=self.requires_replan,
            metadata=dict(self.metadata) if self.metadata else None,
        )

    def copy_with(self, **updates: Any) -> "ActionDirective":
        """Return a new directive with selected fields updated."""

        updated = replace(self, **updates)
        if updated.command_args is not None:
            updated.command_args = dict(updated.command_args)
        if updated.metadata is not None:
            updated.metadata = dict(updated.metadata)
        return updated

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the directive for event publication."""

        data: Dict[str, Any] = {
            "approved": self.approved,
            "requires_replan": self.requires_replan,
        }
        if self.command_name is not None:
            data["command_name"] = self.command_name
        if self.command_args is not None:
            data["command_args"] = dict(self.command_args)
        if self.rationale:
            data["rationale"] = self.rationale
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


class DecisionEngine:
    """Evaluate multiple reasoning paths and select the optimal action."""

    def __init__(self, scoring_fn: Callable[[float, float], float] | None = None):
        self.scoring_fn = scoring_fn or (lambda utility, cost: utility - cost)

    def select_optimal_action(self, plans: Iterable[ActionPlan]) -> Tuple[str, str]:
        """Return the best action and its rationale among ``plans``.

        The default scoring function maximizes ``utility - cost``. The ``rationale``
        of the chosen plan is returned alongside the action. If a plan does not
        provide a rationale, a generic explanation including the computed score is
        produced.
        """

        best_plan: ActionPlan | None = None
        best_score = float("-inf")
        for plan in plans:
            score = self.scoring_fn(plan.utility, plan.cost)
            if score > best_score:
                best_score = score
                best_plan = plan
        if best_plan is None:
            raise ValueError("No plans provided")
        rationale = best_plan.rationale or (
            f"Score={best_score:.3f} (utility={best_plan.utility}, cost={best_plan.cost})"
        )
        return best_plan.action, rationale

    # ------------------------------------------------------------------
    # Action review hooks
    # ------------------------------------------------------------------
    def review_action(
        self,
        agent_id: str,
        command_name: str,
        command_args: Dict[str, str],
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> ActionDirective:
        """Evaluate a proposed action and optionally adjust it.

        The default implementation looks for alternative candidate actions in the
        ``context`` and selects the highest scoring option using
        :meth:`select_optimal_action`. If no candidates are provided the action is
        approved as-is.
        """

        context = context or {}
        plans_data = context.get("candidate_actions") or context.get("plan")
        plans: list[ActionPlan] = []
        if isinstance(plans_data, list):
            for item in plans_data:
                if not isinstance(item, dict):
                    continue
                action = item.get("action") or item.get("command")
                if not isinstance(action, str):
                    continue
                utility = float(item.get("utility", 0.0))
                cost = float(item.get("cost", 0.0))
                rationale = item.get("rationale") or item.get("reason") or ""
                plans.append(
                    ActionPlan(
                        action=action,
                        utility=utility,
                        cost=cost,
                        rationale=str(rationale),
                    )
                )
        if plans:
            best_action, rationale = self.select_optimal_action(plans)
            if best_action != command_name:
                return ActionDirective.approve(
                    best_action,
                    command_args,
                    rationale=f"Replanned action: {rationale}",
                )
        return ActionDirective.approve(command_name, command_args)

    def record_outcome(
        self,
        agent_id: str,
        command_name: str,
        result: Any,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Observe the outcome of a command execution.

        The base implementation performs no action but allows subclasses to learn
        from historical results.
        """

        # Default implementation is a no-op; subclasses may override.
        return None
