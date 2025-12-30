"""Problem solving utilities using associative reasoning and reflection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from backend.reasoning import MultiHopAssociator
from backend.reflection import ReflectionModule


@dataclass
class DivergentConvergentSolver:
    """Generate multiple solution paths and evaluate them.

    The solver first explores divergent solution paths between ``start`` and
    ``goal`` using :class:`~backend.reasoning.MultiHopAssociator`. Each path is
    then scored using :class:`~backend.reflection.ReflectionModule` and the
    highest scoring path is returned.
    """

    associator: MultiHopAssociator
    reflection: ReflectionModule

    def _generate_paths(
        self, start: str, goal: str, strategies: List[str]
    ) -> List[List[str]]:
        """Create possible solution paths via intermediate ``strategies``."""

        paths: List[List[str]] = []
        for strat in strategies:
            first = self.associator.find_path(start, strat)
            second = self.associator.find_path(strat, goal)
            if first and second:
                paths.append(first + second[1:])
        return paths

    def solve(
        self, start: str, goal: str, strategies: List[str]
    ) -> Tuple[List[str], int]:
        """Return the highest scoring path and its score."""

        candidates = []
        for path in self._generate_paths(start, goal, strategies):
            evaluation, _ = self.reflection.reflect(" ".join(path))
            score = int(evaluation.confidence * len(path)) or len(path)
            candidates.append((score, path))
        if not candidates:
            return [], 0
        score, best = max(candidates, key=lambda x: x[0])
        return best, score
