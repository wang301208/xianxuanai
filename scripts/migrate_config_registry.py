"""Populate ConfigRegistry with configuration classes scattered across the codebase.

This script walks the ``autogpt`` package, imports all modules to ensure that
configuration classes are loaded, and registers them with a ``ConfigRegistry``
instance. It can be used during development to verify that all configuration
objects are discoverable by the registry.
"""

from __future__ import annotations

import importlib
import pkgutil

from third_party.autogpt.autogpt.core.configuration import ConfigRegistry


def migrate() -> ConfigRegistry:
    registry = ConfigRegistry()

    # Import all modules under the autogpt package to expose configuration classes
    package = importlib.import_module("autogpt")
    for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            importlib.import_module(module_info.name)
        except Exception:
            # Skip modules that fail to import; they are not required for migration.
            continue

    registry.collect()
    return registry


if __name__ == "__main__":
    reg = migrate()
    print("Registered configurations:", list(reg.to_dict().keys()))
