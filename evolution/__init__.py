"""Compatibility package re-exporting `modules.evolution` as `evolution`.

The codebase contains the implementation under `modules/evolution`, while some
tests and integrations import `evolution.*` directly. This package bridges that
gap by extending the package search path to include `modules/evolution`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore[name-defined]

_MODULES_EVOLUTION = (Path(__file__).resolve().parents[1] / "modules" / "evolution").resolve()
if _MODULES_EVOLUTION.exists():
    __path__.append(str(_MODULES_EVOLUTION))  # type: ignore[name-defined]


class Agent(ABC):
    """Minimal agent base used by `evolution.*` modules."""

    @abstractmethod
    def perform(self, *args, **kwargs):
        raise NotImplementedError


__all__ = ["Agent"]
