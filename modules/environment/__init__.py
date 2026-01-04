"""Environment simulators and loops for embodied cognition."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .registry import (
    get_hardware_registry,
    HardwareEnvironmentRegistry,
    register_service,
    register_rpc_service,
    register_skill_service,
    register_model_service,
    unregister_service,
    report_resource_signal,
    report_service_signal,
    dispatch_task,
    publish_task_result_event,
    subscribe_resource_signals,
    subscribe_service_signals,
    subscribe_service_catalog,
    subscribe_task_dispatch,
    subscribe_task_results,
)

if TYPE_CHECKING:  # pragma: no cover - avoid import-time cycles
    from .simulator import BaseEnvironment as BaseEnvironment
    from .simulator import GridWorldEnvironment as GridWorldEnvironment
    from .loop import ActionPerceptionLoop as ActionPerceptionLoop
    from .environment_adapter import EnvironmentAdapter as EnvironmentAdapter
    from .environment_adapter import EnvironmentAdjustment as EnvironmentAdjustment
    from .environment_adapter import EnvironmentSnapshot as EnvironmentSnapshot


def __getattr__(name: str):  # pragma: no cover - import-time optimisation
    if name in {"BaseEnvironment", "GridWorldEnvironment"}:
        from .simulator import BaseEnvironment, GridWorldEnvironment

        return {"BaseEnvironment": BaseEnvironment, "GridWorldEnvironment": GridWorldEnvironment}[name]
    if name == "ActionPerceptionLoop":
        from .loop import ActionPerceptionLoop as loop_cls

        return loop_cls
    if name in {"EnvironmentAdapter", "EnvironmentAdjustment", "EnvironmentSnapshot"}:
        from .environment_adapter import (
            EnvironmentAdapter,
            EnvironmentAdjustment,
            EnvironmentSnapshot,
        )

        return {
            "EnvironmentAdapter": EnvironmentAdapter,
            "EnvironmentAdjustment": EnvironmentAdjustment,
            "EnvironmentSnapshot": EnvironmentSnapshot,
        }[name]
    raise AttributeError(name)

__all__ = [
    "BaseEnvironment",
    "GridWorldEnvironment",
    "ActionPerceptionLoop",
    "EnvironmentAdapter",
    "EnvironmentAdjustment",
    "EnvironmentSnapshot",
    "get_hardware_registry",
    "HardwareEnvironmentRegistry",
    "register_service",
    "register_rpc_service",
    "register_skill_service",
    "register_model_service",
    "unregister_service",
    "report_resource_signal",
    "report_service_signal",
    "dispatch_task",
    "publish_task_result_event",
    "subscribe_resource_signals",
    "subscribe_service_signals",
    "subscribe_service_catalog",
    "subscribe_task_dispatch",
    "subscribe_task_results",
]
