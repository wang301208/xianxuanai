"""File system watcher for blueprint repository changes.

This module provides a small wrapper around the :mod:`watchdog` package that
watches the blueprint directory for file changes. When a YAML blueprint file is
created or modified the provided callback is invoked with the path to the
changed file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .io import BLUEPRINT_DIR


class _BlueprintEventHandler(FileSystemEventHandler):
    """Internal handler dispatching blueprint change events."""

    def __init__(self, callback: Callable[[Path], None]) -> None:
        self._callback = callback

    def on_created(self, event) -> None:  # type: ignore[override]
        self._handle(event)

    def on_modified(self, event) -> None:  # type: ignore[override]
        self._handle(event)

    def _handle(self, event) -> None:
        if getattr(event, "is_directory", False):
            return
        path = Path(getattr(event, "src_path"))
        if path.suffix.lower() in {".yaml", ".yml"}:
            self._callback(path)


class BlueprintWatcher:
    """Watch the blueprint directory for changes.

    Parameters
    ----------
    callback:
        Function invoked with the :class:`pathlib.Path` of the changed
        blueprint file.
    """

    def __init__(self, callback: Callable[[Path], None]) -> None:
        self._callback = callback
        self._observer = Observer()
        handler = _BlueprintEventHandler(callback)
        self._observer.schedule(handler, str(BLUEPRINT_DIR), recursive=False)

    def start(self) -> None:
        """Start monitoring the blueprint directory."""
        self._observer.daemon = True
        self._observer.start()

    def stop(self) -> None:
        """Stop monitoring and wait for the observer thread to finish."""
        self._observer.stop()
        self._observer.join()


__all__ = ["BlueprintWatcher"]
