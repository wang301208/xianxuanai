"""Bootstrap utilities for launching BrainSimulationSystem backends.

This module exposes helpers shared by the AutoGPT agent stack and the
BrainSimulationSystem CLI so both environments rely on the same
``BrainSimulationConfig`` contract. The helpers surface convenience
wrappers for resolving configuration overrides, constructing
``BrainSimulationSystemAdapter`` instances, and retrieving the underlying
simulation object when needed by local tooling such as visualisers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from third_party.autogpt.autogpt.core.brain.config import BrainSimulationConfig

from . import BrainBackendInitError, BrainSimulationSystemAdapter


@dataclass(slots=True)
class BrainSimulationBootstrap:
    """Container bundling the resolved config and instantiated adapter."""

    config: BrainSimulationConfig
    adapter: BrainSimulationSystemAdapter

    @property
    def brain(self) -> Any:
        """Expose the wrapped BrainSimulation instance if present."""

        return getattr(self.adapter, "_brain", None)

    def backend_payload(self) -> Dict[str, Any]:
        """Return the adapter payload derived from the configuration."""

        return self.config.to_backend_payload()


def _merge_override_dict(
    base: Mapping[str, Any] | None, extra: Mapping[str, Any] | None
) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base or {})
    if not extra:
        return merged
    for key, value in extra.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _merge_override_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_brain_simulation_config(
    config: BrainSimulationConfig | Mapping[str, Any] | None = None,
    *,
    profile: str | None = None,
    stage: str | None = None,
    config_file: str | Path | None = None,
    overrides_json: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    timestep_ms: float | None = None,
    auto_background: bool | None = None,
    use_environment_defaults: bool = True,
) -> BrainSimulationConfig:
    """Normalise configuration inputs into a ``BrainSimulationConfig`` instance."""

    if isinstance(config, BrainSimulationConfig):
        resolved = config
    else:
        if isinstance(config, Mapping):
            base_overrides: Mapping[str, Any] | None = config
        else:
            base_overrides = None
        if use_environment_defaults:
            resolved = BrainSimulationConfig.from_env()
        else:
            resolved = BrainSimulationConfig()
        if base_overrides:
            resolved = resolved.model_copy(update=dict(base_overrides))

    update_payload: Dict[str, Any] = {}
    if profile is not None:
        update_payload["profile"] = profile
    if stage is not None:
        update_payload["stage"] = stage
    if config_file is not None:
        update_payload["config_file"] = str(config_file)
    if overrides_json is not None:
        update_payload["overrides_json"] = overrides_json
    if timestep_ms is not None:
        update_payload["timestep_ms"] = float(timestep_ms)
    if auto_background is not None:
        update_payload["auto_background"] = bool(auto_background)

    if update_payload:
        resolved = resolved.model_copy(update=update_payload)

    if overrides:
        merged = _merge_override_dict(resolved.overrides, overrides)
        resolved = resolved.model_copy(update={"overrides": merged})

    return resolved


def bootstrap_brain_simulation(
    config: BrainSimulationConfig | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> BrainSimulationBootstrap:
    """Instantiate a ``BrainSimulationSystemAdapter`` using shared config logic."""

    resolved_config = resolve_brain_simulation_config(config, **kwargs)
    try:
        adapter = BrainSimulationSystemAdapter(resolved_config)
    except BrainBackendInitError:
        raise
    except Exception as exc:  # pragma: no cover - safety net for unexpected errors
        raise BrainBackendInitError(
            "Unexpected error initialising BrainSimulationSystem"
        ) from exc
    return BrainSimulationBootstrap(config=resolved_config, adapter=adapter)


__all__ = [
    "BrainSimulationBootstrap",
    "bootstrap_brain_simulation",
    "resolve_brain_simulation_config",
]
