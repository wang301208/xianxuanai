from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from pathlib import Path
from types import ModuleType
from typing import Dict

from watchdog.events import FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

import logging


logger = logging.getLogger(__name__)


class AlgorithmPluginLoader(FileSystemEventHandler):
    """Discover, load and hot-reload algorithm plugins."""

    def __init__(self, group: str = "autogpt.algorithms") -> None:
        self.group = group
        self.modules: Dict[str, ModuleType] = {}
        self._backups: Dict[str, ModuleType] = {}
        self.observer = Observer()

    def load_plugins(self) -> None:
        """Load all entry point plugins for the configured group."""
        for ep in entry_points().select(group=self.group):
            module = ep.load()
            self.modules[ep.name] = module
            self._backups[ep.name] = module
            path = Path(module.__file__).resolve().parent
            self.observer.schedule(self, str(path), recursive=False)

    # Watchdog API ---------------------------------------------------------
    def on_modified(self, event: FileModifiedEvent) -> None:  # type: ignore[override]
        path = Path(event.src_path)
        for name, module in self.modules.items():
            if Path(module.__file__).resolve() == path:
                self._reload(name, module)
                break

    # Internal helpers -----------------------------------------------------
    def _reload(self, name: str, module: ModuleType) -> None:
        try:
            new_module = importlib.reload(module)
            self.modules[name] = new_module
            self._backups[name] = new_module
        except (ImportError, ModuleNotFoundError, SyntaxError) as err:
            logger.exception("Failed to reload plugin %s", name)
            # rollback to last good version
            self.modules[name] = self._backups[name]

    def start(self) -> None:
        self.observer.start()

    def stop(self) -> None:
        self.observer.stop()
        self.observer.join()
