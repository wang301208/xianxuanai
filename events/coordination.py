"""Compatibility module that re-exports :mod:`modules.events.coordination`."""

from modules.events.coordination import (  # type: ignore[import-not-found]
    IterationEvent,
    TaskDispatchEvent,
    TaskStatus,
    TaskStatusEvent,
)

__all__ = [
    "TaskStatus",
    "TaskDispatchEvent",
    "TaskStatusEvent",
    "IterationEvent",
]
