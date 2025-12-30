"""Simple chain-of-thought based planner."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel

from .schema import Task, TaskStatus


class PlanResult(BaseModel):
    """Result returned by :class:`ReasoningPlanner`.

    Attributes:
        reasoning: The intermediate reasoning steps produced while
            evaluating the task.
        next_step: The next step or action that should be taken.
    """

    reasoning: List[str]
    next_step: str


class ReasoningPlanner:
    """Planner that performs a simple chain-of-thought reasoning process.

    The implementation is intentionally lightweight and deterministic so it can
    be unit tested without relying on an external language model.  It iterates
    over the task's ready criteria and records the thought process before
    proposing the next step.
    """

    def plan(self, task: Task) -> PlanResult:
        """Generate reasoning steps and the next plan for a given task.

        Parameters:
            task: The task for which a plan should be created.

        Returns:
            A :class:`PlanResult` containing the reasoning chain and the
            proposed next step.
        """

        reasoning: List[str] = [f"Objective: {task.objective}"]

        if task.ready_criteria:
            for idx, criterion in enumerate(task.ready_criteria, start=1):
                reasoning.append(f"Step {idx}: {criterion}")
        else:
            reasoning.append("Step 1: No specific ready criteria.")

        next_step = task.objective
        if task.context.status == TaskStatus.DONE:
            next_step = ""

        return PlanResult(reasoning=reasoning, next_step=next_step)

