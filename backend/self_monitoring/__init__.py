"""Self-monitoring utilities with lazy exports."""

from importlib import import_module
from typing import Any

__all__ = [
    "DiagnosticReport",
    "GuardianDecision",
    "HealthAlert",
    "HealthMonitor",
    "ModuleRemediationPlan",
    "RecoveryDecision",
    "RemediationGuardian",
    "RemediationManager",
    "RemediationPatch",
    "RemediationResult",
    "SelfMonitoringSystem",
    "SensorReading",
    "StepReport",
    "SystemDiagnostics",
]

_EXPORT_MAP = {
    "DiagnosticReport": ("backend.self_monitoring.diagnostics", "DiagnosticReport"),
    "SystemDiagnostics": ("backend.self_monitoring.diagnostics", "SystemDiagnostics"),
    "GuardianDecision": ("backend.self_monitoring.guardian", "GuardianDecision"),
    "RemediationGuardian": ("backend.self_monitoring.guardian", "RemediationGuardian"),
    "HealthAlert": ("backend.self_monitoring.health", "HealthAlert"),
    "HealthMonitor": ("backend.self_monitoring.health", "HealthMonitor"),
    "SensorReading": ("backend.self_monitoring.health", "SensorReading"),
    "ModuleRemediationPlan": ("backend.self_monitoring.remediation", "ModuleRemediationPlan"),
    "RemediationManager": ("backend.self_monitoring.remediation", "RemediationManager"),
    "RemediationPatch": ("backend.self_monitoring.remediation", "RemediationPatch"),
    "RemediationResult": ("backend.self_monitoring.remediation", "RemediationResult"),
    "RecoveryDecision": ("backend.self_monitoring.monitor", "RecoveryDecision"),
    "SelfMonitoringSystem": ("backend.self_monitoring.monitor", "SelfMonitoringSystem"),
    "StepReport": ("backend.self_monitoring.monitor", "StepReport"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value
    return value
