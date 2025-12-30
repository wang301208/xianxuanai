"""Central configuration orchestrator used across services."""
from __future__ import annotations

from copy import deepcopy
from threading import RLock
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel

from .providers import ConfigProvider

T = TypeVar("T", bound=BaseModel)
Listener = Callable[["ConfigurationHub"], None]


class ConfigNotFoundError(KeyError):
    """Raised when a configuration section is missing."""


class ConfigurationHub:
    """Aggregate configuration from multiple providers with typed accessors."""

    def __init__(
        self,
        providers: Iterable[ConfigProvider],
        eager_load: bool = True,
    ) -> None:
        self._providers = tuple(providers)
        if not self._providers:
            raise ValueError("ConfigurationHub requires at least one provider")
        self._lock = RLock()
        self._raw: Dict[str, Any] = {}
        self._model_cache: Dict[Tuple[Optional[str], Type[BaseModel]], BaseModel] = {}
        self._listeners: list[Listener] = []
        if eager_load:
            self.reload()

    def reload(self) -> None:
        """Reload configuration from all providers in declaration order."""

        merged: Dict[str, Any] = {}
        for provider in self._providers:
            payload = provider.load()
            if not isinstance(payload, MutableMapping):
                raise TypeError(
                    f"Provider {provider.__class__.__name__} must return a mapping"
                )
            merged = _deep_merge(merged, payload)
        with self._lock:
            self._raw = merged
            self._model_cache.clear()
        self._notify_listeners()

    def get_raw(self) -> Dict[str, Any]:
        """Return a deep copy of the current configuration payload."""

        with self._lock:
            return deepcopy(self._raw)

    def get(
        self,
        path: Optional[str] = None,
        *,
        model: Optional[Type[T]] = None,
        default: Optional[Any] = None,
    ) -> Any | T:
        """Fetch configuration by dotted `path` returning optional typed model."""

        with self._lock:
            target = self._raw if path is None else _resolve_path(self._raw, path)
        if target is None:
            if default is not None:
                return default
            raise ConfigNotFoundError(path or "root")
        if model is None:
            return deepcopy(target)
        cache_key = (path, model)
        with self._lock:
            cached = self._model_cache.get(cache_key)
        if cached is not None:
            return cached
        instance = _build_model(model, target)
        with self._lock:
            self._model_cache[cache_key] = instance
        return instance

    def add_listener(self, callback: Listener) -> None:
        """Register a listener invoked after reload."""

        self._listeners.append(callback)

    def _notify_listeners(self) -> None:
        for callback in self._listeners:
            try:
                callback(self)
            except Exception:  # pragma: no cover - defensive guard
                pass


def _resolve_path(payload: Mapping[str, Any], path: str) -> Any:
    cursor: Any = payload
    for piece in path.split("."):
        if not isinstance(cursor, Mapping) or piece not in cursor:
            return None
        cursor = cursor[piece]
    return cursor


def _deep_merge(base: Dict[str, Any], incoming: Mapping[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in incoming.items():
        if (
            key in result
            and isinstance(result[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)  # type: ignore[index]
        else:
            result[key] = deepcopy(value)
    return result


def _build_model(model: Type[T], payload: Any) -> T:
    if hasattr(model, "model_validate"):
        return model.model_validate(payload)  # type: ignore[attr-defined]
    return model.parse_obj(payload)  # type: ignore[attr-defined]
