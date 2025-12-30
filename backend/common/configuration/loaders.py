"""Helper utilities to bootstrap project-wide configuration hub."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

from .hub import ConfigurationHub
from .providers import EnvVarConfigProvider, YamlConfigProvider

T = TypeVar("T", bound=BaseModel)

_DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
_ENV_CONFIG_DIR_KEY = "AUTOAI_CONFIG_DIR"
_ENV_PREFIX_KEY = "AUTOAI_CONFIG_PREFIX"

_default_hub: Optional[ConfigurationHub] = None


def get_hub() -> ConfigurationHub:
    """Return the global configuration hub (lazy singleton)."""

    global _default_hub
    if _default_hub is None:
        providers = _build_default_providers()
        _default_hub = ConfigurationHub(providers)
    return _default_hub


def get_settings(path: Optional[str] = None, *, model: Optional[Type[T]] = None) -> Any | T:
    """Shortcut to fetch configuration from the default hub."""

    return get_hub().get(path, model=model)


def reload_settings() -> None:
    """Force refresh of the global configuration hub."""

    get_hub().reload()


def _build_default_providers():
    config_root = Path(os.environ.get(_ENV_CONFIG_DIR_KEY, str(_DEFAULT_CONFIG_DIR)))
    providers = []
    if config_root.exists():
        providers.append(YamlConfigProvider(config_root))
    else:
        raise FileNotFoundError(
            f"Configuration directory {config_root} was not found; set {_ENV_CONFIG_DIR_KEY}"
        )
    prefix = os.environ.get(_ENV_PREFIX_KEY, "APP_")
    providers.append(EnvVarConfigProvider(prefix=prefix))
    return providers
