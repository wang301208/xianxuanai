from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol


@dataclass
class SelfState:
    """Container for the agent's internal self state."""

    current_goal: str = ""
    memory_pointer: int = 0
    action_history: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "current_goal": self.current_goal,
            "memory_pointer": self.memory_pointer,
            "recent_actions": self.action_history[-5:],
        }


class GlobalWorkspace:
    """Simple global workspace for coordinating agent subsystems."""

    def __init__(self) -> None:
        self.state = SelfState()
        self._observers: List[WorkspaceObserver] = []
        self._feedback_history: List[Any] = []

    def update_state(
        self,
        goal: str | None = None,
        memory_pointer: int | None = None,
        action: str | None = None,
    ) -> None:
        if goal is not None:
            self.state.current_goal = goal
        if memory_pointer is not None:
            self.state.memory_pointer = memory_pointer
        if action is not None:
            self.state.action_history.append(action)

    def get_context(self) -> Dict[str, Any]:
        """Return a dictionary representation of the current self state."""
        return self.state.as_dict()

    def reflect(self, predicted_action: str, actual_action: str) -> None:
        """Very small reflection mechanism comparing expected and actual actions."""
        if predicted_action != actual_action:
            self.state.action_history.append(
                f"reflection: expected {predicted_action} but executed {actual_action}"
            )

    # ------------------------------------------------------------------
    # Observer integration
    # ------------------------------------------------------------------
    def register_observer(self, observer: "WorkspaceObserver") -> None:
        """Register a workspace observer that can critique agent thoughts."""

        self._observers.append(observer)

    def broadcast(
        self,
        thought: Dict[str, Any] | str,
        *,
        context: Dict[str, Any] | None = None,
    ) -> List[Any]:
        """Broadcast ``thought`` to all observers and collect feedback."""

        if isinstance(thought, dict):
            serialized = json.dumps(thought, ensure_ascii=False)
        else:
            serialized = str(thought)
        merged_context = {**self.get_context(), **(context or {})}
        feedback: List[Any] = []
        for observer in self._observers:
            signal = observer.evaluate(serialized, merged_context)
            if signal:
                feedback.append(signal)
        if feedback:
            self._feedback_history.extend(feedback)
        return feedback

    def feedback_history(self) -> List[Any]:
        """Return collected feedback from observers."""

        return list(self._feedback_history)


class WorkspaceObserver(Protocol):
    name: str

    def evaluate(self, thought: str, context: Dict[str, Any]) -> Any:
        """Return feedback for ``thought`` given workspace ``context``."""
