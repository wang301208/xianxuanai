"""Compatibility package re-exporting `modules.monitoring` as `monitoring`."""

from __future__ import annotations

from importlib import import_module

_mod = import_module("modules.monitoring")

ActionLogger = getattr(_mod, "ActionLogger")
RealTimeMetricsCollector = getattr(_mod, "RealTimeMetricsCollector")
MetricEvent = getattr(_mod, "MetricEvent")
BottleneckDetector = getattr(_mod, "BottleneckDetector")
PerformanceDiagnoser = getattr(_mod, "PerformanceDiagnoser")
DiagnosticIssue = getattr(_mod, "DiagnosticIssue")
RollingAnomalyModel = getattr(_mod, "RollingAnomalyModel")
WorkspaceMessage = getattr(_mod, "WorkspaceMessage")
GlobalWorkspace = getattr(_mod, "GlobalWorkspace")
global_workspace = getattr(_mod, "global_workspace")

__all__ = [
    "ActionLogger",
    "RealTimeMetricsCollector",
    "MetricEvent",
    "BottleneckDetector",
    "PerformanceDiagnoser",
    "DiagnosticIssue",
    "RollingAnomalyModel",
    "WorkspaceMessage",
    "GlobalWorkspace",
    "global_workspace",
]
