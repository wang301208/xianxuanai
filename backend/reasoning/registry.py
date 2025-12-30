"""Registry for sharing reasoning components across the process."""

from __future__ import annotations

from typing import Optional

from .symbolic import SymbolicReasoner
from .commonsense import CommonsenseValidator
from .causal import KnowledgeGraphCausalReasoner


_SYMBOLIC: SymbolicReasoner | None = None
_COMMONSENSE: CommonsenseValidator | None = None
_CAUSAL: KnowledgeGraphCausalReasoner | None = None


def set_symbolic_reasoner(reasoner: SymbolicReasoner) -> None:
    global _SYMBOLIC
    _SYMBOLIC = reasoner


def get_symbolic_reasoner() -> Optional[SymbolicReasoner]:
    return _SYMBOLIC


def require_symbolic_reasoner() -> SymbolicReasoner:
    reasoner = get_symbolic_reasoner()
    if reasoner is None:
        raise RuntimeError("Symbolic reasoner has not been initialised.")
    return reasoner


def set_commonsense_validator(validator: CommonsenseValidator) -> None:
    global _COMMONSENSE
    _COMMONSENSE = validator


def get_commonsense_validator() -> Optional[CommonsenseValidator]:
    return _COMMONSENSE


def require_commonsense_validator() -> CommonsenseValidator:
    validator = get_commonsense_validator()
    if validator is None:
        raise RuntimeError("Commonsense validator has not been initialised.")
    return validator


def set_causal_reasoner(reasoner: KnowledgeGraphCausalReasoner) -> None:
    global _CAUSAL
    _CAUSAL = reasoner


def get_causal_reasoner() -> Optional[KnowledgeGraphCausalReasoner]:
    return _CAUSAL


def require_causal_reasoner() -> KnowledgeGraphCausalReasoner:
    reasoner = get_causal_reasoner()
    if reasoner is None:
        raise RuntimeError("Causal reasoner has not been initialised.")
    return reasoner
