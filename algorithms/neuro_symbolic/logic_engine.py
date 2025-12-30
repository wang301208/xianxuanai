"""Simple neuro-symbolic logic engine.

This module bridges neural network outputs with symbolic reasoning. Given the
output of a neural model (expressed as a mapping of symbols to booleans or
truthy values) and a set of symbolic rules, the :class:`LogicEngine` evaluates
those rules to derive additional facts.
"""

from __future__ import annotations

from typing import Any, Dict


class LogicEngine:
    """Evaluate symbolic rules against neural network outputs.

    Parameters
    ----------
    rules:
        Mapping of rule names to boolean expressions using symbols present in
        the neural network output.
    """

    def __init__(self, rules: Dict[str, str] | None = None) -> None:
        self.rules = rules or {}

    def evaluate(self, nn_output: Dict[str, Any]) -> Dict[str, bool]:
        """Run symbolic reasoning based on neural network predictions.

        Parameters
        ----------
        nn_output:
            Mapping of symbol names to values produced by a neural network.

        Returns
        -------
        Dict[str, bool]
            The evaluated truth values for each rule.
        """

        context = {key: bool(value) for key, value in nn_output.items()}
        results: Dict[str, bool] = {}
        for name, expr in self.rules.items():
            try:
                results[name] = bool(eval(expr, {"__builtins__": {}}, context))
            except NameError as exc:
                raise ValueError(f"Unknown symbol in rule '{expr}': {exc}") from exc
        return results

