"""Compatibility layer for neuromorphic hardware backends.

This module historically implemented dedicated neuromorphic hardware
integrations.  The canonical implementation now lives under
``BrainSimulationSystem.core.backends``.  We keep this file as a thin
wrapper so existing imports continue to function while delegating all
logic to the unified backend manager.
"""

from __future__ import annotations

from BrainSimulationSystem.core.backends import (
    BaseNeuromorphicInterface,
    BrainScaleSBackend,
    HardwareConfig,
    HardwarePlatform as NeuromorphicPlatform,
    ModelHardwareTranslator,
    NeuromorphicBackend,
    NeuromorphicBackendManager,
    NeuromorphicEvent,
    SpikeEvent,
    WeightMapping,
    create_neuromorphic_backend_manager,
    create_neuromorphic_interface,
    detect_available_hardware,
    get_backend,
    native_backend_factories,
)

__all__ = [
    "NeuromorphicPlatform",
    "NeuromorphicBackend",
    "NeuromorphicEvent",
    "NeuromorphicBackendManager",
    "HardwareConfig",
    "SpikeEvent",
    "WeightMapping",
    "BaseNeuromorphicInterface",
    "ModelHardwareTranslator",
    "BrainScaleSBackend",
    "create_neuromorphic_interface",
    "create_neuromorphic_backend_manager",
    "detect_available_hardware",
    "get_backend",
    "native_backend_factories",
]
