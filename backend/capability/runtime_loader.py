from __future__ import annotations

"""Runtime loading and unloading of capability modules.

This module defines :class:`RuntimeModuleManager` which relies on the
:mod:`module_registry` to instantiate capability modules on demand and keeps
track of which modules are currently active.  Modules can be requested or
released dynamically while the system is running, allowing agents to adapt to
new goals without restarting the entire application.
"""

import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .module_registry import available_modules, get_module

try:  # Import the common module interface if available
    from modules.interface import ModuleInterface
except Exception:  # pragma: no cover - modules package may not be present
    ModuleInterface = None  # type: ignore

try:  # Optional – the manager can operate without an EventBus
    from events import EventBus  # type: ignore
except Exception:  # pragma: no cover - events module may not be available
    EventBus = None  # type: ignore


class RuntimeModuleManager:
    """Manage dynamic loading/unloading of capability modules."""

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self._loaded: Dict[str, Any] = {}
        self._loaded_since: Dict[str, float] = {}
        self._last_used_ts: Dict[str, float] = {}
        self._bus = event_bus
        if self._bus is not None:
            self._bus.subscribe("module.request", self._on_request)
            self._bus.subscribe("module.release", self._on_release)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    async def _on_request(self, event: Dict[str, Any]) -> None:
        name = event.get("module")
        if not isinstance(name, str):
            return
        self.load(name)

    async def _on_release(self, event: Dict[str, Any]) -> None:
        name = event.get("module")
        if not isinstance(name, str):
            return
        try:
            self.unload(name)
        except KeyError:
            return

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self, name: str, *args, **kwargs) -> Any:
        """Load ``name`` if not already loaded and return the module instance."""
        now = time.time()
        if name in self._loaded:
            self._last_used_ts[name] = now
            if self._bus:
                try:
                    self._bus.publish("module.used", {"time": now, "module": name, "cached": True})
                except Exception:
                    pass
            return self._loaded[name]
        start = time.time()
        module = get_module(name, *args, **kwargs)

        # Resolve dependencies for ModuleInterface implementations
        if ModuleInterface is not None and isinstance(module, ModuleInterface):
            for dep in getattr(module, "dependencies", []):
                self.load(dep)
            module.initialize()

        now = time.time()
        self._loaded[name] = module
        self._loaded_since[name] = now
        self._last_used_ts[name] = now
        if self._bus:
            try:
                load_seconds = now - start
            except Exception:
                load_seconds = None
            try:
                payload: Dict[str, Any] = {"time": now, "module": name}
                if load_seconds is not None:
                    payload["load_seconds"] = float(load_seconds)
                self._bus.publish("module.loaded", payload)
            except Exception:
                pass
            try:
                self._bus.publish("module.used", {"time": now, "module": name, "cached": False})
            except Exception:
                pass
        return module

    def unload(self, name: str) -> None:
        """Unload a previously loaded module."""
        now = time.time()
        module = self._loaded.pop(name)
        loaded_since = float(self._loaded_since.pop(name, now) or now)
        self._last_used_ts.pop(name, None)
        # Gracefully shut down modules. If a module implements ModuleInterface
        # we call its explicit ``shutdown`` hook; otherwise fall back to common
        # cleanup method names.
        if ModuleInterface is not None and isinstance(module, ModuleInterface):
            try:
                module.shutdown()
            except Exception:  # pragma: no cover - best effort
                pass
        else:
            for method in ("shutdown", "close", "stop"):
                func = getattr(module, method, None)
                if callable(func):
                    try:
                        func()
                    except Exception:  # pragma: no cover - best effort
                        pass
                    break
        if self._bus:
            try:
                payload: Dict[str, Any] = {"time": now, "module": name}
                payload["loaded_seconds"] = max(0.0, float(now) - float(loaded_since))
                self._bus.publish("module.unloaded", payload)
            except Exception:
                pass

    def ensure(self, required: Iterable[str]) -> Dict[str, Any]:
        """Ensure ``required`` modules are loaded (without unloading others)."""

        return self.update(required, prune=False)

    def update(self, required: Iterable[str], *, prune: bool = True) -> Dict[str, Any]:
        """Ensure ``required`` modules are loaded and optionally drop others.

        Parameters
        ----------
        required:
            Iterable of module names needed for upcoming work. Only names that
            appear in :func:`available_modules` are considered.
        prune:
            When True (default) unload modules not in ``required``. When False,
            only loads missing modules, leaving other loaded modules untouched.
        """
        available = set(available_modules())
        required_list: List[str] = []
        needed: List[str] = []
        needed_set: set[str] = set()
        for item in required or []:
            if not isinstance(item, str):
                continue
            token = item.strip()
            if not token:
                continue
            required_list.append(token)
            if token in available and token not in needed_set:
                needed.append(token)
                needed_set.add(token)

        effective_prune = bool(prune)
        if effective_prune and required_list and not needed_set:
            # If callers request only unknown/disabled modules, avoid unloading
            # everything as a side-effect (common when inputs are free-form).
            effective_prune = False

        if effective_prune:
            for name in list(self._loaded.keys()):
                if name not in needed_set:
                    self.unload(name)

        loaded = {name: self.load(name) for name in needed}
        if self._bus:
            try:
                self._bus.publish(
                    "module.requirements",
                    {
                        "time": time.time(),
                        "required": list(required_list),
                        "needed": list(needed),
                        "loaded": sorted(self._loaded.keys()),
                        "prune": bool(effective_prune),
                    },
                )
            except Exception:
                pass
        return loaded

    def loaded_modules(self) -> List[str]:
        """Return a list of names for currently loaded modules."""
        return list(self._loaded.keys())

    def get_loaded(self, name: str) -> Any | None:
        """Return the loaded module instance if present (without loading new)."""

        token = str(name or "").strip()
        if not token:
            return None
        module = self._loaded.get(token)
        if module is None:
            return None
        now = time.time()
        self._last_used_ts[token] = now
        if self._bus:
            try:
                self._bus.publish("module.used", {"time": now, "module": token, "cached": True})
            except Exception:
                pass
        return module
