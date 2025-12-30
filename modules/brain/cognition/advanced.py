from __future__ import annotations

"""Simplified higher level cognitive architecture.

This module implements an ``AdvancedCognitiveSystem`` composed of four small
sub‑components.  The classes are **not** intended to model human cognition
accurately; they merely provide a light‑weight API that can be exercised in the
unit tests.  The components are:

``WorkingMemorySystem``
    Bounded capacity store with FIFO replacement policy.
``ExecutiveControlNetwork``
    Provides a couple of toy executive functions operating on simple data
    structures.
``DecisionMakingSystem``
    Demonstrates dual‑process and reinforcement‑learning style decisions.
``MetacognitiveSystem``
    Returns a confidence estimate for a given decision.
"""

from dataclasses import dataclass, field
from typing import Dict, List


class WorkingMemorySystem:
    """Finite capacity working memory."""

    def __init__(self, capacity: int = 4) -> None:
        self.capacity = capacity
        self._items: List[str] = []

    def store(self, item: str) -> None:
        """Store ``item`` respecting the capacity limit.

        The implementation uses a FIFO replacement policy – once the capacity
        is exceeded the oldest item is discarded.  This behaviour is sufficient
        for the unit tests which only check that the number of retained items
        never exceeds ``capacity``.
        """

        self._items.append(item)
        if len(self._items) > self.capacity:
            # Drop the oldest element
            self._items.pop(0)

    def retrieve(self) -> List[str]:
        """Return a copy of the stored items."""

        return list(self._items)


class ExecutiveControlNetwork:
    """Very small collection of executive control operations."""

    def conflict_monitoring(self, stimuli: List[str]) -> bool:
        """Detect conflicting stimuli.

        We model conflict simply as duplicate entries in ``stimuli``.  The
        method returns ``True`` when a conflict is detected, ``False``
        otherwise.
        """

        return len(set(stimuli)) < len(stimuli)

    def cognitive_flexibility(self, task_history: List[str]) -> str:
        """Decide whether to repeat the last task or switch to a new one."""

        if len(task_history) < 2:
            return "repeat"
        return "switch" if task_history[-1] != task_history[-2] else "repeat"

    def response_inhibition(self, impulse: bool) -> bool:
        """Return ``False`` when an impulse should be inhibited."""

        return not impulse

    def updating_working_memory(self, wm: WorkingMemorySystem, item: str) -> List[str]:
        """Insert ``item`` into ``wm`` and return the updated contents."""

        wm.store(item)
        return wm.retrieve()


@dataclass
class DecisionMakingSystem:
    """Toy decision making algorithms."""

    q_table: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def dual_process(self, options: Dict[str, float]) -> str:
        """Fast heuristic when an option is obviously best, otherwise deliberate."""

        for option, value in options.items():
            if value >= 0.8:
                return option
        # deliberate: pick the option with highest value
        return max(options, key=options.get)

    def reinforcement_learning(self, state: str, rewards: Dict[str, float]) -> str:
        """Choose the action with the highest reward and record it in ``q_table``."""

        action = max(rewards, key=rewards.get)
        self.q_table.setdefault(state, {})[action] = rewards[action]
        return action


class MetacognitiveSystem:
    """Return a basic confidence score for a decision."""

    def assess_confidence(self, decision: str, evidence_strength: float) -> float:
        """Clamp ``evidence_strength`` into [0, 1] and return it as confidence."""

        return max(0.0, min(1.0, evidence_strength))


class AdvancedCognitiveSystem:
    """Facade orchestrating the cognitive sub‑components."""

    def __init__(self, memory_capacity: int = 4) -> None:
        self.working_memory = WorkingMemorySystem(capacity=memory_capacity)
        self.executive = ExecutiveControlNetwork()
        self.decision_maker = DecisionMakingSystem()
        self.metacognition = MetacognitiveSystem()

    # -- Working memory -----------------------------------------------------
    def store_in_memory(self, item: str) -> None:
        self.working_memory.store(item)

    def retrieve_memory(self) -> List[str]:
        return self.working_memory.retrieve()

    # -- Executive control --------------------------------------------------
    def conflict_monitoring(self, stimuli: List[str]) -> bool:
        return self.executive.conflict_monitoring(stimuli)

    def cognitive_flexibility(self, task_history: List[str]) -> str:
        return self.executive.cognitive_flexibility(task_history)

    def response_inhibition(self, impulse: bool) -> bool:
        return self.executive.response_inhibition(impulse)

    def update_working_memory(self, item: str) -> List[str]:
        return self.executive.updating_working_memory(self.working_memory, item)

    # -- Decision making ----------------------------------------------------
    def make_decision(self, options: Dict[str, float], model: str = "dual") -> str:
        if model == "dual":
            return self.decision_maker.dual_process(options)
        if model == "rl":
            return self.decision_maker.reinforcement_learning("default", options)
        raise ValueError(f"unknown decision model: {model}")

    # -- Metacognition ------------------------------------------------------
    def assess_confidence(self, decision: str, evidence_strength: float) -> float:
        return self.metacognition.assess_confidence(decision, evidence_strength)


__all__ = [
    "WorkingMemorySystem",
    "ExecutiveControlNetwork",
    "DecisionMakingSystem",
    "MetacognitiveSystem",
    "AdvancedCognitiveSystem",
]
