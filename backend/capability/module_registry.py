from __future__ import annotations

from typing import Any, Callable, Dict, List

_REGISTRY: Dict[str, Callable[..., Any]] = {}
_DISABLED: set[str] = set()


def register_module(name: str, factory: Callable[..., Any]) -> None:
    """Register a module factory under ``name``."""
    _REGISTRY[name] = factory


def disable_module(name: str) -> None:
    """Disable a module without unregistering it."""

    token = str(name or "").strip()
    if token:
        _DISABLED.add(token)


def enable_module(name: str) -> None:
    """Re-enable a previously disabled module."""

    token = str(name or "").strip()
    if token:
        _DISABLED.discard(token)


def is_module_enabled(name: str) -> bool:
    token = str(name or "").strip()
    if not token:
        return False
    if token not in _REGISTRY:
        return False
    return token not in _DISABLED


def disabled_modules() -> List[str]:
    return sorted(_DISABLED)


def unregister_module(name: str) -> None:
    """Remove a module from the registry entirely."""

    token = str(name or "").strip()
    if not token:
        return
    _REGISTRY.pop(token, None)
    _DISABLED.discard(token)


def get_module(name: str, *args, allow_disabled: bool = False, **kwargs) -> Any:
    """Instantiate a registered module."""
    if name not in _REGISTRY:
        raise KeyError(f"Module '{name}' is not registered")
    if not allow_disabled and name in _DISABLED:
        raise KeyError(f"Module '{name}' is disabled")
    return _REGISTRY[name](*args, **kwargs)


def available_modules(*, include_disabled: bool = False) -> List[str]:
    """Return a list of registered module names.

    By default disabled modules are excluded so callers can treat this as the
    "currently usable" capability set.
    """

    if include_disabled:
        return list(_REGISTRY.keys())
    return [name for name in _REGISTRY.keys() if name not in _DISABLED]


def combine_modules(names: List[str]) -> List[Any]:
    """Instantiate multiple modules by name."""
    return [get_module(name) for name in names]


__all__ = [
    "register_module",
    "unregister_module",
    "disable_module",
    "enable_module",
    "is_module_enabled",
    "disabled_modules",
    "get_module",
    "available_modules",
    "combine_modules",
]
