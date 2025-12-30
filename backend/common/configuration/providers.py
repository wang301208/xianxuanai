"""Config providers supplying key/value dictionaries to the ConfigurationHub."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict
import json
import os

import yaml


class ConfigProvider(ABC):
    """Abstract configuration provider returning a nested dictionary."""

    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """Return configuration payload."""


class YamlConfigProvider(ConfigProvider):
    """Load configuration from a YAML file or an entire directory."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Config path {self._path} not found")

    def load(self) -> Dict[str, Any]:
        if self._path.is_dir():
            data: Dict[str, Any] = {}
            for file in sorted(self._path.glob("*.yml")) + sorted(
                self._path.glob("*.yaml")
            ):
                data = _deep_merge_dicts(data, _load_yaml(file))
            return data
        return _load_yaml(self._path)


class EnvVarConfigProvider(ConfigProvider):
    """Read configuration overrides from environment variables.

    Keys use a prefix (default `APP_`) and `separator` to express nesting.
    Example: `APP_logging__level=DEBUG` becomes `{"logging": {"level": "DEBUG"}}`.
    Values try to decode JSON so booleans or lists can be expressed easily.
    """

    def __init__(self, prefix: str = "APP_", separator: str = "__") -> None:
        self._prefix = prefix
        self._separator = separator

    def load(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for key, raw_value in os.environ.items():
            if not key.startswith(self._prefix):
                continue
            path = key[len(self._prefix) :].lower().split(self._separator)
            value: Any = _coerce_env_value(raw_value)
            cursor = payload
            for part in path[:-1]:
                cursor = cursor.setdefault(part, {})  # type: ignore[assignment]
            cursor[path[-1]] = value
        return payload


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"YAML config at {path} must produce a dictionary")
        return loaded


def _deep_merge_dicts(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in incoming.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            base[key] = _deep_merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def _coerce_env_value(value: str) -> Any:
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        return value
