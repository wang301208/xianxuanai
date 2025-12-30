"""Interaction flow for the arena with optional exploration mode."""
from __future__ import annotations

from typing import List, Protocol

from ..execution.planner import Planner
from ..founder_agent.analytics import Analytics


class AgentProtocol(Protocol):
    """Protocol that arena agents are expected to follow."""

    def act(self, task: str) -> dict:
        """Execute a task and return metric events."""

    def generate_goal(self, reward: float) -> str | None:
        """Optionally generate a new goal based on an intrinsic reward."""


class Arena:
    """Coordinate agent interaction and optionally explore new goals."""

    def __init__(self, planner: Planner | None = None, analytics: Analytics | None = None) -> None:
        self.planner = planner or Planner()
        self.analytics = analytics or Analytics()

    def interact(self, agent: AgentProtocol, goal: str, exploration: bool = False) -> None:
        """Run interaction steps for a given goal.

        Parameters
        ----------
        agent:
            The agent participating in the arena.
        goal:
            High level objective for the agent.
        exploration:
            When ``True``, the agent may propose new goals based on the
            intrinsic reward computed from the interaction so far.
        """
        tasks: List[str] = self.planner.decompose(goal)
        idx = 0
        while idx < len(tasks):
            task = tasks[idx]
            result = agent.act(task)
            if isinstance(result, dict):
                self.analytics.handle_event(result)
            if exploration:
                reward = self.analytics.get_intrinsic_reward()
                new_goal = agent.generate_goal(reward)
                if new_goal:
                    tasks.extend(self.planner.decompose(new_goal))
            idx += 1
