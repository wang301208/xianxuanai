"""Compatibility module that re-exports :mod:`modules.events.client`."""

from modules.events.client import EventClient  # type: ignore[import-not-found]

__all__ = ["EventClient"]
