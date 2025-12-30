"""Adapter for evolution module implementing the common ModuleInterface."""

from __future__ import annotations

try:  # pragma: no cover - support both repo-root and `modules/` on sys.path
    from modules.interface import ModuleInterface
except ModuleNotFoundError:  # pragma: no cover
    from interface import ModuleInterface


class EvolutionModule(ModuleInterface):
    """Expose evolution capabilities via ModuleInterface."""

    dependencies: list[str] = []

    def __init__(self) -> None:
        self.initialized = False

    def initialize(self) -> None:  # pragma: no cover - trivial
        self.initialized = True

    def shutdown(self) -> None:  # pragma: no cover - trivial
        self.initialized = False
