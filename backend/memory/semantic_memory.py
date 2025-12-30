"""Semantic memory storing structured facts."""

from __future__ import annotations

from typing import Dict, Optional


class SemanticMemory:
    """Dictionary based semantic memory."""

    def __init__(self) -> None:
        self._facts: Dict[str, str] = {}

    def add(self, key: str, value: str) -> None:
        """Add a semantic fact."""
        self._facts[key] = value

    def get(self, key: str) -> Optional[str]:
        """Retrieve a fact by key."""
        return self._facts.get(key)

    def all(self) -> Dict[str, str]:
        """Return a copy of all known facts."""
        return dict(self._facts)

    def clear(self) -> None:
        """Remove all facts."""
        self._facts.clear()

    def remove(self, key: str) -> None:
        """Remove a single fact if it exists."""

        self._facts.pop(key, None)
