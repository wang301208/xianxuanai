"""Compatibility package exposing `backend.knowledge` as top-level `knowledge`.

Some modules import `knowledge.UnifiedKnowledgeBase` directly. The implementation
in this repository lives under `backend/knowledge`, so we re-export it here to
avoid requiring an external installation step.
"""

from __future__ import annotations

from backend.knowledge import *  # type: ignore
from backend.knowledge import __all__ as _backend_all

__all__ = list(_backend_all)

