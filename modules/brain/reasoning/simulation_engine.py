"""Scenario simulation engine integrating world model and rule repository."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, List

State = Dict[str, Any]
RuleFunc = Callable[[State], State | List[State]]
Rule = Dict[str, RuleFunc]
Rules = Dict[str, Rule]


class ScenarioSimulationEngine:
    """Engine for multi-step scenario simulation.

    The engine combines a *world model* represented as an arbitrary state
    dictionary with a repository of rules mapping action names to forward and
    backward transition functions. It can simulate sequences of actions in the
    forward direction and also perform reverse simulation using the backward
    rules. Rules may yield multiple states to support branching scenarios.

    Parameters
    ----------
    world_model:
        Optional default state used when no ``initial_state`` is provided.
    rules:
        Optional pre-populated rule repository.
    """

    def __init__(self, world_model: State | None = None, rules: Rules | None = None) -> None:
        self.world_model: State = world_model or {}
        self.rules: Rules = rules or {}

    def add_rule(self, action: str, forward: RuleFunc, backward: RuleFunc) -> None:
        """Register ``forward`` and ``backward`` rules for ``action``."""

        self.rules[action] = {"forward": forward, "backward": backward}

    def simulate_scenario(
        self,
        initial_state: State | None,
        actions: List[str],
        reverse: bool = False,
    ) -> List[List[State]]:
        """Simulate ``actions`` starting from ``initial_state``.

        Returns a history list where each element contains the possible states
        after the corresponding step. The first element represents the initial
        state. When ``reverse`` is true the actions are processed in reverse
        order using their backward rules.
        """

        state = deepcopy(initial_state) if initial_state is not None else deepcopy(self.world_model)
        states: List[State] = [state]
        history: List[List[State]] = [deepcopy(states)]
        sequence = list(reversed(actions)) if reverse else actions
        for action in sequence:
            rule = self.rules.get(action)
            if not rule:
                history.append(states)
                continue
            fn = rule["backward"] if reverse else rule["forward"]
            next_states: List[State] = []
            for s in states:
                result = fn(deepcopy(s))
                if isinstance(result, list):
                    next_states.extend(result)
                else:
                    next_states.append(result)
            states = next_states
            history.append(deepcopy(states))
        return history

    def predict_outcome(self, initial_state: State | None, actions: List[str]) -> List[State]:
        """Return the final state(s) after simulating ``actions``."""

        return self.simulate_scenario(initial_state, actions)[-1]


__all__ = ["ScenarioSimulationEngine"]
