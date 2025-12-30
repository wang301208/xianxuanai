"""Analogical reasoning via simple structural mapping and vector retrieval."""

from __future__ import annotations

from collections import Counter
from math import sqrt
from typing import Dict, Tuple

Vector = Dict[str, int]
Structure = Dict[str, str]


class AnalogicalReasoner:
    """Minimal analogical reasoner using bag-of-words vector search.

    Knowledge is stored per domain as pairs of a structured representation and
    a free-text description. During transfer the description is compared to the
    target task via cosine similarity and roles from the best matching structure
    are mapped onto the target structure.
    """

    def __init__(self) -> None:
        self._knowledge: Dict[str, list[Tuple[Structure, str, Vector]]] = {}

    # ------------------ internal helpers ------------------
    def _vectorize(self, text: str) -> Vector:
        return Counter(text.lower().split())

    def _cosine(self, a: Vector, b: Vector) -> float:
        shared = set(a) & set(b)
        num = sum(a[w] * b[w] for w in shared)
        denom = sqrt(sum(v * v for v in a.values())) * sqrt(sum(v * v for v in b.values()))
        return num / denom if denom else 0.0

    # ------------------ public API ------------------
    def add_knowledge(self, domain: str, structure: Structure, description: str) -> None:
        """Store ``structure`` and ``description`` under ``domain``."""

        vec = self._vectorize(description)
        self._knowledge.setdefault(domain, []).append((structure, description, vec))

    def transfer_knowledge(
        self, source_domain: str, target_description: str, target_structure: Structure
    ) -> Dict[str, str] | None:
        """Transfer best matching knowledge from ``source_domain`` to target task.

        Parameters
        ----------
        source_domain:
            Domain containing source knowledge.
        target_description:
            Free-text description of the target task.
        target_structure:
            Structural representation of the target task using the same role keys
            as in the source domain.

        Returns
        -------
        mapping:
            A dictionary mapping source entities to target entities, or ``None``
            if no knowledge exists for the domain.
        """

        candidates = self._knowledge.get(source_domain)
        if not candidates:
            return None
        target_vec = self._vectorize(target_description)
        best: Tuple[Structure, str, Vector] | None = None
        best_score = -1.0
        for structure, _desc, vec in candidates:
            score = self._cosine(vec, target_vec)
            if score > best_score:
                best = (structure, _desc, vec)
                best_score = score
        assert best is not None
        source_structure = best[0]
        return {source_structure[k]: target_structure.get(k, source_structure[k]) for k in source_structure}


__all__ = ["AnalogicalReasoner"]
