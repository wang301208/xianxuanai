"""Monitoring utilities for AutoGPT."""

from .action_logger import ActionLogger
from .storage import TimeSeriesStorage
from .system_metrics import SystemMetricsCollector
from .metrics_collector import MetricsCollector
from .performance_monitor import PerformanceMonitor, email_alert, dashboard_alert
from .calibration import PerformanceCalibrator, CalibrationProfile, CalibrationRecord
from .auto_optimizer import AutoOptimizer
from .reflection import Reflection
from .global_workspace import GlobalWorkspace, WorkspaceMessage, global_workspace
from .resource_scheduler import ResourceScheduler
from .brain_state import record_memory_hit, get_memory_hits
from .multi_metric_monitor import MultiMetricMonitor
from .evaluation import EvaluationMetrics


def create_brain_app(*args, **kwargs):
    """Lazily import and instantiate the brain state FastAPI application."""

    from .brain_state import create_app

    return create_app(*args, **kwargs)

__all__ = [
    "TimeSeriesStorage",
    "ActionLogger",
    "SystemMetricsCollector",
    "MetricsCollector",
    "PerformanceMonitor",
    "PerformanceCalibrator",
    "CalibrationProfile",
    "CalibrationRecord",
    "email_alert",
    "dashboard_alert",
    "AutoOptimizer",
    "Reflection",
    "GlobalWorkspace",
    "WorkspaceMessage",
    "global_workspace",
    "ResourceScheduler",
    "create_brain_app",
    "record_memory_hit",
    "get_memory_hits",
    "MultiMetricMonitor",
    "EvaluationMetrics",
]
