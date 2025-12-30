from __future__ import annotations

from typing import Any, Callable, Iterable, List, Tuple


class Reflection:
    """Evaluate candidate solutions and select the best option."""

    def __init__(self, score_fn: Callable[[Any], float]) -> None:
        self.score_fn = score_fn

    def evaluate(self, candidates: Iterable[Any]) -> List[Tuple[Any, float]]:
        """Return ``(candidate, score)`` pairs sorted by score."""

        scored: List[Tuple[Any, float]] = []
        for candidate in candidates:
            try:
                score = self.score_fn(candidate)
            except Exception:  # noqa: BLE001
                continue
            scored.append((candidate, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def select_best(self, candidates: Iterable[Any]) -> Any | None:
        """Return the candidate with the highest score or ``None``."""

        scored = self.evaluate(candidates)
        return scored[0][0] if scored else None
