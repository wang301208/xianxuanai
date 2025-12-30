from __future__ import annotations

import importlib
import logging
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Iterable

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


class ModernDependencyManager:
    """Install and verify Python package dependencies.

    Uses ``importlib`` to check for packages and ``pip`` to install them when
    necessary. Version specifiers are honored when provided.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self._logger = logger or logging.getLogger(__name__)

    def ensure(self, requirement: str) -> bool:
        """Ensure that a package requirement is installed.

        Args:
            requirement: A requirement string, e.g. ``"package>=1.0,<2.0"``.

        Returns:
            ``True`` if the package is installed and satisfies the requirement or
            was installed successfully, otherwise ``False``.
        """

        req = Requirement(requirement)
        if self._is_satisfied(req):
            return True

        self._logger.warning(
            "Required package '%s' is not installed or doesn't satisfy %s."
            " Attempting installation.",
            req.name,
            req.specifier if req.specifier else "",
        )
        return self._install(requirement)

    def ensure_all(self, requirements: Iterable[str]) -> None:
        for req in requirements:
            self.ensure(req)

    def _is_satisfied(self, req: Requirement) -> bool:
        try:
            installed_version = version(req.name)
        except PackageNotFoundError:
            return False
        spec: SpecifierSet = req.specifier
        if spec and not spec.contains(installed_version, prereleases=True):
            return False
        try:
            importlib.import_module(req.name)
            return True
        except ImportError:
            return False

    def _install(self, requirement: str) -> bool:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", requirement],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return True
        except subprocess.CalledProcessError:
            self._logger.warning("Failed to install package '%s'", requirement)
            return False
