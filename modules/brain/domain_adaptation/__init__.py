"""Domain adaptation framework for brain module.

This package provides a generic :class:`DomainAdapter` interface and helper
utilities for dynamically loading domain specific adapters.  Adapters map the
brain's generic interface to domain specific tools or knowledge bases.
"""

from .core import DomainAdapter, DomainAdapterManager, register_adapter, load_adapter, available_adapters

__all__ = [
    "DomainAdapter",
    "DomainAdapterManager",
    "register_adapter",
    "load_adapter",
    "available_adapters",
]
