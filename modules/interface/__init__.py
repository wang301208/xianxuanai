from __future__ import annotations

"""Common interface for AutoGPT modules."""

from abc import ABC, abstractmethod
from typing import List


class ModuleInterface(ABC):
    """Abstract base class for all dynamic modules.

    Modules can declare other modules they depend on via the ``dependencies``
    attribute. The :class:`RuntimeModuleManager` uses this information to load
    modules in the correct order and to call the lifecycle hooks.
    """

    #: Names of modules that must be loaded before this module initializes.
    dependencies: List[str] = []

    @abstractmethod
    def initialize(self) -> None:
        """Perform any setup after all dependencies have loaded."""

    @abstractmethod
    def shutdown(self) -> None:
        """Gracefully release resources before the module is unloaded."""
