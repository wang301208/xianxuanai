"""Ethical reasoning engine for evaluating planned actions.

This module provides an :class:`EthicalReasoningEngine` that can be
configured with a library of ethical rules and value weights.  Each rule
specifies a condition on an action, the ethical value it relates to and a
suggestion for remediation.  The engine evaluates an action against all
rules and aggregates a weighted score of any violations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional


@dataclass
class EthicalRule:
    """Representation of a single ethical rule.

    Attributes
    ----------
    condition:
        A callable that receives the action description and returns ``True``
        when the rule is violated.
    value:
        Name of the ethical value the rule is associated with (e.g.
        ``"nonmaleficence"``).
    suggestion:
        Guidance provided when the rule is violated.
    """

    condition: Callable[[str], bool]
    value: str
    suggestion: str


class EthicalReasoningEngine:
    """Evaluate actions against a set of ethical rules.

    Parameters
    ----------
    rules:
        Iterable of :class:`EthicalRule` objects representing the rule
        library.
    value_weights:
        Optional mapping from value names to their relative weights when
        scoring violations.  Unspecified values default to a weight of ``1``.
    """

    def __init__(
        self,
        rules: Optional[Iterable[EthicalRule]] = None,
        value_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.rules: List[EthicalRule] = list(rules) if rules else []
        self.value_weights: Dict[str, float] = value_weights or {}

    def add_rule(self, rule: EthicalRule) -> None:
        """Add an :class:`EthicalRule` to the library."""

        self.rules.append(rule)

    def evaluate_action(self, action: str) -> Dict[str, object]:
        """Evaluate ``action`` and return compliance and suggestions.

        The returned dictionary contains:

        ``compliant``:
            ``True`` when no rules are violated.
        ``score``:
            Aggregated weight of violated rules.
        ``suggestions``:
            List of remediation suggestions for violated rules.
        """

        description = str(action)
        compliant = True
        score = 0.0
        suggestions: List[str] = []

        for rule in self.rules:
            if rule.condition(description):
                compliant = False
                weight = self.value_weights.get(rule.value, 1)
                score += weight
                suggestions.append(rule.suggestion)

        return {
            "compliant": compliant,
            "score": score,
            "suggestions": suggestions,
        }
