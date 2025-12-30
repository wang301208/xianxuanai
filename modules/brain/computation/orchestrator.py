"""Orchestrator for executing algorithms by name."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, Type

from algorithms.base import Algorithm


class ComputationOrchestrator:
    """Utility class to load and execute algorithms from the ``algorithms`` package."""

    def __init__(self, root_package: str = "algorithms") -> None:
        self.root_package = root_package

    def _find_module(self, name: str):
        """Locate the module containing the algorithm.

        Args:
            name: The module name (without package prefix) to search for.

        Returns:
            The imported module containing the algorithm implementation.

        Raises:
            ValueError: If no matching module is found.
        """
        package = importlib.import_module(self.root_package)
        for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            if module_name.endswith(f".{name}"):
                return importlib.import_module(module_name)
        raise ValueError(f"Algorithm '{name}' not found")

    def execute_algorithm(self, name: str, data: Any, params: Dict[str, Any] | None = None) -> Any:
        """Execute an algorithm by name and return the result.

        Args:
            name: The algorithm's module name (e.g. ``"bubble_sort"``).
            data: Primary data input passed as the first argument to the algorithm.
            params: Optional dictionary of additional keyword arguments.

        Returns:
            The result produced by the algorithm's ``execute`` method.

        Raises:
            ValueError: If the algorithm cannot be located or does not define a subclass of
                :class:`~algorithms.base.Algorithm`.
        """
        module = self._find_module(name)
        algorithm_cls: Type[Algorithm] | None = None
        for attr in module.__dict__.values():
            if isinstance(attr, type) and issubclass(attr, Algorithm) and attr is not Algorithm:
                algorithm_cls = attr
                break
        if algorithm_cls is None:
            raise ValueError(f"No Algorithm subclass found for '{name}'")
        algorithm = algorithm_cls()
        params = params or {}
        return algorithm.execute(data, **params)
