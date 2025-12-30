from __future__ import annotations

"""Solvers that mix symbolic rules with probabilistic reasoning."""

from typing import Callable, Dict, Iterable, List, Tuple
import json


class RuleProbabilisticSolver:
    """Apply rule-based inference with associated probabilities."""

    def __init__(self, rules: Dict[str, List[Tuple[str, float]]]):
        """Initialize with mapping of antecedents to (conclusion, probability) tuples."""
        self.rules = rules

    def infer(self, statement: str, evidence: Iterable[str]) -> Tuple[str, float]:
        """Return the most probable conclusion given ``statement`` and ``evidence``."""
        best: Tuple[str, float] | None = None
        for antecedent, conclusions in self.rules.items():
            if antecedent == statement or antecedent in evidence:
                for conclusion, prob in conclusions:
                    if not best or prob > best[1]:
                        best = (conclusion, prob)
        return best if best else (statement, 1.0)


class NeuroSymbolicSolver:
    """Combine neural predictions with symbolic rules.

    The solver first queries a neural model for candidate conclusions and their
    probabilities. It then merges these predictions with a symbolic knowledge
    base, which can override or introduce additional conclusions.

    Example
    -------
    >>> def model(statement, evidence):
    ...     return {"B": 0.2}
    >>> solver = NeuroSymbolicSolver(model, {"A": "C"})
    >>> solver.infer("A", [])
    ("C", 1.0)
    """

    def __init__(
        self,
        neural_model: Callable[[str, Iterable[str]], Dict[str, float]] | None = None,
        knowledge_base: Dict[str, str] | None = None,
    ) -> None:
        self.neural_model = neural_model or (lambda _s, _e: {})
        self.knowledge_base = knowledge_base or {}

    # ------------------------------------------------------------------
    # Loading utilities
    # ------------------------------------------------------------------
    def load_neural_model(self, weights_path: str) -> None:
        """Load neural model weights from ``weights_path``.

        The file should contain a JSON mapping from statements to dictionaries
        of ``{conclusion: probability}``.
        """

        with open(weights_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        def model(statement: str, _evidence: Iterable[str]) -> Dict[str, float]:
            return data.get(statement, {})

        self.neural_model = model

    def load_symbolic_kb(self, kb_path: str) -> None:
        """Load symbolic rules from ``kb_path``.

        The file must contain a JSON mapping ``{antecedent: conclusion}``.
        """

        with open(kb_path, "r", encoding="utf-8") as f:
            self.knowledge_base = json.load(f)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def infer(self, statement: str, evidence: Iterable[str]) -> Tuple[str, float]:
        """Return the merged conclusion for ``statement``.

        Neural predictions are first gathered. Any applicable symbolic rule will
        introduce its conclusion with probability ``1.0`` (overriding weaker
        neural suggestions). The highest probability conclusion is returned.
        """

        predictions: Dict[str, float] = dict(self.neural_model(statement, evidence))
        for antecedent, conclusion in self.knowledge_base.items():
            if antecedent == statement or antecedent in evidence:
                prob = predictions.get(conclusion, 0.0)
                predictions[conclusion] = max(prob, 1.0)

        if predictions:
            conclusion, probability = max(predictions.items(), key=lambda p: p[1])
            return conclusion, float(probability)
        return statement, 1.0
