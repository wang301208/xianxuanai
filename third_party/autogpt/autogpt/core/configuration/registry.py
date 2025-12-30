from __future__ import annotations

"""Registry for SystemConfiguration and SystemSettings instances.

This registry provides a central place to collect configuration objects used
throughout the system. It supports loading values from environment variables,
YAML configuration files and explicit overrides (e.g. from command line
arguments).
"""

from pathlib import Path
from typing import Any, Dict

import yaml

from .schema import (
    SystemConfiguration,
    SystemSettings,
    _update_user_config_from_env,
    deep_update,
)


class ConfigRegistry:
    """Collect and manage configuration objects."""

    def __init__(self) -> None:
        self._configs: Dict[str, SystemConfiguration] = {}
        self._settings: Dict[str, SystemSettings] = {}

    # ------------------------------------------------------------------
    # Registration & collection
    # ------------------------------------------------------------------
    def register(self, instance: SystemConfiguration | SystemSettings) -> None:
        """Register a configuration or settings instance."""
        if isinstance(instance, SystemConfiguration):
            key = instance.__class__.__name__
            self._configs[key] = instance
        elif isinstance(instance, SystemSettings):
            key = instance.__class__.__name__
            self._settings[key] = instance
        else:
            raise TypeError("Unsupported type for registry")

    def collect(self) -> None:
        """Automatically collect subclasses of configuration/settings."""
        for cls in SystemConfiguration.__subclasses__():
            try:
                instance = cls.from_env()
            except Exception:
                try:
                    instance = cls()
                except Exception:
                    continue
            self.register(instance)

        for cls in SystemSettings.__subclasses__():
            try:
                instance = cls()
            except Exception:
                continue
            self.register(instance)

    # ------------------------------------------------------------------
    # Loading and overrides
    # ------------------------------------------------------------------
    def load_from_env(self) -> None:
        """Update registered objects from environment variables."""
        for name, conf in list(self._configs.items()):
            data = conf.dict()
            data.update(_update_user_config_from_env(conf))
            self._configs[name] = conf.__class__.parse_obj(data)

        for name, st in list(self._settings.items()):
            data = st.dict()
            data.update(_update_user_config_from_env(st))
            self._settings[name] = st.__class__.parse_obj(data)

    def load_from_yaml(self, path: str | Path) -> None:
        """Load overrides from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self.apply_overrides(data)

    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply overrides (e.g. from command line)."""
        for key, value in overrides.items():
            if key in self._configs:
                inst = self._configs[key]
                merged = deep_update(inst.dict(), value)
                self._configs[key] = inst.__class__.parse_obj(merged)
            elif key in self._settings:
                inst = self._settings[key]
                merged = deep_update(inst.dict(), value)
                self._settings[key] = inst.__class__.parse_obj(merged)

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------
    def get(self, name: str) -> SystemConfiguration | SystemSettings | None:
        return self._configs.get(name) or self._settings.get(name)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        data: Dict[str, Dict[str, Any]] = {
            name: inst.dict() for name, inst in self._configs.items()
        }
        data.update({name: inst.dict() for name, inst in self._settings.items()})
        return data
