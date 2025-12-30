from __future__ import annotations

"""Protocol definitions for reasoning plugins."""

from typing import Iterable, Protocol, Tuple


class KnowledgeSource(Protocol):
    """Source of information for reasoning."""

    def query(self, statement: str) -> Iterable[str]:
        """Return evidence relevant to ``statement``."""


class Solver(Protocol):
    """Performs inference given a statement and supporting evidence."""

    def infer(self, statement: str, evidence: Iterable[str]) -> Tuple[str, float]:
        """Return a conclusion and its associated probability/confidence."""


class CausalReasoner(Protocol):
    """Infer causal relationships between events or concepts."""

    def check_causality(self, cause: str, effect: str) -> Tuple[bool, Iterable[str]]:
        """Return whether ``cause`` leads to ``effect`` and the supporting path."""


class CounterfactualReasoner(Protocol):
    """Evaluate the outcome of alternative scenarios."""

    def evaluate_counterfactual(self, cause: str, effect: str) -> str:
        """Return an explanation of ``effect`` if ``cause`` were different."""
