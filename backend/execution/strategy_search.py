from __future__ import annotations

import random
from typing import Any, Callable, Iterable, List


class StrategySearch:
    """Generate multiple solution paths using random and heuristic strategies."""

    def __init__(
        self,
        heuristics: Iterable[Callable[[Any], List[str]]] | None = None,
        seed: int | None = None,
    ) -> None:
        self.heuristics = list(heuristics or [])
        self._rand = random.Random(seed)

    def random_paths(self, options: List[str], count: int) -> List[List[str]]:
        """Return ``count`` random permutations of *options*."""

        paths: List[List[str]] = []
        for _ in range(max(count, 0)):
            items = list(options)
            self._rand.shuffle(items)
            paths.append(items)
        return paths

    def heuristic_paths(self, context: Any) -> List[List[str]]:
        """Generate paths using registered heuristic callables."""

        paths: List[List[str]] = []
        for func in self.heuristics:
            try:
                path = func(context)
            except Exception:  # noqa: BLE001
                continue
            if path:
                paths.append(list(path))
        return paths

    def search(
        self, options: List[str], count: int, context: Any | None = None
    ) -> List[List[str]]:
        """Combine heuristic and random strategies to generate candidate paths."""

        paths = []
        if context is not None:
            paths.extend(self.heuristic_paths(context))
        remaining = count - len(paths)
        if remaining > 0:
            paths.extend(self.random_paths(options, remaining))
        return paths
