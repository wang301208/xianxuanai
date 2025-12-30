"""
Factory and registry utilities for cognitive modules.

This module decouples the core brain simulation from concrete subsystem
implementations.  Components can be registered programmatically or provided
through configuration by specifying a dotted import path.  The factory will
instantiate the requested module while attempting common constructor
signatures, so existing components that expect ``(network, config)`` continue
to work out of the box.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Protocol, runtime_checkable

try:  # Python <3.11 compatibility for Protocol runtime checks
    from typing import Self
except ImportError:  # pragma: no cover - fallback for older runtimes
    Self = Any  # type: ignore[misc]


@runtime_checkable
class CognitiveModuleProtocol(Protocol):
    """
    Minimal protocol implemented by cognitive subsystems.

    Components typically expose a ``process`` method that consumes an input
    dictionary and returns structured results.  The protocol remains minimal to
    avoid over-constraining downstream modules while still supporting type
    checking in the orchestrator.
    """

    def process(self, inputs: Dict[str, Any]) -> Any:  # pragma: no cover - structural typing only
        ...



# Default implementations keyed by the logical component name.  Consumers may
# override entries via ``ModuleFactory.register`` or by providing a ``class``
# value inside the component configuration.
DEFAULT_COMPONENTS: Dict[str, str] = {
    "perception": "BrainSimulationSystem.models.perception.PerceptionProcess",
    "attention": "BrainSimulationSystem.models.attention.AttentionProcess",
    "memory": "BrainSimulationSystem.models.memory.MemoryProcess",
    "decision": "BrainSimulationSystem.models.decision.DecisionProcess",
}


def resolve_symbol(dotted_path: str) -> Any:
    """Import and return the symbol referenced by ``dotted_path``."""

    module_path, _, symbol = dotted_path.rpartition(".")
    if not module_path:
        raise ImportError(f"Invalid dotted path '{dotted_path}'")
    module = importlib.import_module(module_path)
    try:
        return getattr(module, symbol)
    except AttributeError as exc:  # pragma: no cover - defensive branch
        raise ImportError(f"Module '{module_path}' has no attribute '{symbol}'") from exc


def _is_signature_mismatch(error: TypeError) -> bool:
    """Return True if the TypeError likely stems from argument mismatch."""

    message = str(error)
    signature_markers = [
        "positional argument",
        "unexpected keyword argument",
        "required positional argument",
        "takes from",
        "takes",
    ]
    return any(marker in message for marker in signature_markers)


def instantiate_module(
    target: Any,
    network: Optional[Any],
    config: Dict[str, Any],
) -> Any:
    """
    Instantiate ``target`` with common constructor signatures.

    The order of attempts is:
        1. (network, config)
        2. (network,)
        3. (config,)
        4. ()

    If instantiation raises a ``TypeError`` for reasons unrelated to argument
    mismatch the exception is re-raised to preserve the original failure
    behaviour.
    """

    attempts: Iterable[tuple] = []
    if network is not None:
        attempts = (
            (network, config),
            (network,),
            (config,),
            (),
        )
    else:
        attempts = (
            (config,),
            (),
        )

    errors: list[str] = []
    for args in attempts:
        try:
            return target(*args)
        except TypeError as exc:
            if not _is_signature_mismatch(exc):
                raise
            errors.append(f"{target}(*{args!r}): {exc}")
            continue
    joined = "; ".join(errors) or "no valid constructor signature found"
    raise TypeError(f"Failed to instantiate module {target}: {joined}")


@dataclass(frozen=True)
class ModuleContainer:
    """
    Lightweight container that exposes the instantiated modules while keeping a
    reference to the factory that produced them.
    """

    factory: "ModuleFactory"
    components: Dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.components[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.components.get(key, default)


class ModuleFactory:
    """Builds cognitive modules based on registry entries and configuration."""

    def __init__(
        self,
        network: Optional[Any],
        *,
        registry: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._network = network
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        self._class_map: Dict[str, str] = dict(DEFAULT_COMPONENTS)
        if registry:
            for key, value in registry.items():
                self.register(key, value)

    def register(self, component: str, implementation: Any) -> None:
        """
        Register or override the implementation for ``component``.

        Args:
            component: Logical component name (e.g. ``"attention"``).
            implementation: Either a dotted import path or a callable that
                returns a component instance when invoked with ``(network,
                config)``.
        """

        if callable(implementation):  # custom builder
            setattr(self, f"_builder_{component}", implementation)
        elif isinstance(implementation, str):
            self._class_map[component] = implementation
        else:  # pragma: no cover - defensive check
            raise TypeError(
                f"Unsupported implementation for '{component}': {implementation!r}"
            )

    def unregister(self, component: str) -> None:
        """Remove ``component`` from the registry and any custom builder."""

        self._class_map.pop(component, None)
        builder_attr = f"_builder_{component}"
        if hasattr(self, builder_attr):
            delattr(self, builder_attr)

    def registry_snapshot(self) -> Dict[str, Any]:
        """Return the currently registered components and their targets."""

        snapshot: Dict[str, Any] = dict(self._class_map)
        for name in dir(self):
            if name.startswith("_builder_"):
                component_name = name[len("_builder_") :]
                snapshot[component_name] = getattr(self, name)
        return snapshot

    def _get_builder(self, component: str) -> Optional[Any]:
        return getattr(self, f"_builder_{component}", None)

    def build(self, component: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """Instantiate the requested ``component`` using the provided config."""

        component_cfg = dict(config or {})
        override = component_cfg.pop("class", None) or component_cfg.pop("target", None)
        builder = self._get_builder(component)
        if builder is not None:
            return builder(self._network, component_cfg)

        class_path = override or self._class_map.get(component)
        if not class_path:
            raise ValueError(f"No implementation registered for component '{component}'")

        try:
            target = resolve_symbol(class_path)
        except ImportError as exc:  # pragma: no cover - configuration error path
            self._logger.error("Failed to resolve component '%s': %s", component, exc)
            raise

        return instantiate_module(target, self._network, component_cfg)

    def build_many(self, mapping: Dict[str, Dict[str, Any]]) -> ModuleContainer:
        """Build multiple components in one pass and return a container."""

        components = {}
        for name, cfg in mapping.items():
            components[name] = self.build(name, cfg)
        return ModuleContainer(factory=self, components=components)

    def clone_for(self, network: Optional[Any]) -> Self:
        """Create a new factory bound to ``network`` with identical registry."""

        clone = ModuleFactory(network, logger=self._logger)
        clone._class_map = dict(self._class_map)
        for name in dir(self):
            if name.startswith("_builder_"):
                setattr(clone, name, getattr(self, name))
        return clone

