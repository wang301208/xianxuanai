from __future__ import annotations

"""Core classes and utilities for domain adaptation."""

from abc import ABC, abstractmethod
from importlib import import_module
from typing import Dict, Type


class DomainAdapter(ABC):
    """Abstract base class for mapping generic queries to domain tools."""

    @abstractmethod
    def process(self, query: str) -> str:
        """Process ``query`` using domain specific resources."""
        raise NotImplementedError


# Registry of available domain adapters
_REGISTRY: Dict[str, Type[DomainAdapter]] = {}


def register_adapter(name: str, adapter: Type[DomainAdapter]) -> None:
    """Register an adapter class under ``name``."""
    _REGISTRY[name] = adapter


def load_adapter(name: str) -> DomainAdapter:
    """Load and instantiate the adapter registered under ``name``.

    If the adapter hasn't been registered yet this function attempts to import
    ``modules.brain.domain_adaptation.<name>`` which is expected to register the
    adapter on import.  This allows adapters to be provided as optional
    plugins."""
    if name not in _REGISTRY:
        try:
            import_module(f"modules.brain.domain_adaptation.{name}")
        except ModuleNotFoundError as exc:  # pragma: no cover - error path
            raise KeyError(f"Unknown domain adapter: {name}") from exc
    adapter_cls = _REGISTRY.get(name)
    if adapter_cls is None:
        raise KeyError(f"Unknown domain adapter: {name}")
    return adapter_cls()


def available_adapters() -> Dict[str, Type[DomainAdapter]]:
    """Return mapping of registered adapter names to classes."""
    return dict(_REGISTRY)


class DomainAdapterManager:
    """Manage selection and usage of domain adapters."""

    def __init__(self, domain: str | None = None) -> None:
        self._adapter: DomainAdapter | None = None
        if domain is not None:
            self.set_domain(domain)

    def set_domain(self, domain: str) -> None:
        """Load and activate ``domain`` adapter."""
        self._adapter = load_adapter(domain)

    def process(self, query: str) -> str:
        """Process ``query`` with the active adapter."""
        if self._adapter is None:
            raise RuntimeError("No domain adapter selected")
        return self._adapter.process(query)
