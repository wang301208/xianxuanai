"""Automatic optimization utilities for monitoring module.

The :class:`AutoOptimizer` adjusts resource allocation or model parameters
based on simple threshold strategies. Optimization decisions and their
observed effects are recorded in the time-series storage for later analysis.
"""
from __future__ import annotations

from typing import Dict, Tuple, Any

from .performance_monitor import PerformanceMonitor
from .storage import TimeSeriesStorage


class AutoOptimizer:
    """Adjust resources or parameters based on monitoring metrics.

    The optimizer uses basic threshold-based heuristics to tweak resource
    allocation limits (``cpu_limit`` and ``memory_limit``) as well as arbitrary
    model parameters. All optimization decisions are logged to the provided
    :class:`~monitoring.storage.TimeSeriesStorage` instance for offline
    analysis.
    """

    def __init__(
        self,
        monitor: PerformanceMonitor,
        storage: TimeSeriesStorage,
        *,
        cpu_threshold: float | None = None,
        memory_threshold: float | None = None,
        accuracy_threshold: float | None = None,
        adjustment_factor: float = 0.1,
        parameter_bounds: Dict[str, Tuple[float, float]] | None = None,
    ) -> None:
        self.monitor = monitor
        self.storage = storage
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.accuracy_threshold = accuracy_threshold
        self.adjustment_factor = adjustment_factor
        self.resource_allocation: Dict[str, float] = {
            "cpu_limit": 1.0,
            "memory_limit": 1.0,
        }
        self.parameter_bounds = parameter_bounds or {}
        self.model_params: Dict[str, float] = {
            name: (bounds[0] + bounds[1]) / 2 for name, bounds in self.parameter_bounds.items()
        }

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def step(self) -> None:
        """Run one optimization step.

        This checks the latest metrics from :class:`PerformanceMonitor` and
        adjusts resource allocations or model parameters when thresholds are
        exceeded. Each adjustment is recorded together with the observed metric
        values before and after the change.
        """

        self._optimize_resources()
        self._optimize_parameters()

    # ------------------------------------------------------------------
    # resource optimization
    # ------------------------------------------------------------------
    def _optimize_resources(self) -> None:
        if self.cpu_threshold is not None:
            cpu_usage = self.monitor.cpu_usage()
            if cpu_usage > self.cpu_threshold:
                before = self.resource_allocation["cpu_limit"]
                after = before * (1 + self.adjustment_factor)
                self.resource_allocation["cpu_limit"] = after
                self.storage.store(
                    "optimization",
                    {
                        "metric": "cpu",
                        "decision": {
                            "resource": "cpu_limit",
                            "before": before,
                            "after": after,
                        },
                        "effect": {
                            "usage_before": cpu_usage,
                            "usage_after": self.monitor.cpu_usage(),
                        },
                    },
                )
        if self.memory_threshold is not None:
            mem_usage = self.monitor.memory_usage()
            if mem_usage > self.memory_threshold:
                before = self.resource_allocation["memory_limit"]
                after = before * (1 + self.adjustment_factor)
                self.resource_allocation["memory_limit"] = after
                self.storage.store(
                    "optimization",
                    {
                        "metric": "memory",
                        "decision": {
                            "resource": "memory_limit",
                            "before": before,
                            "after": after,
                        },
                        "effect": {
                            "usage_before": mem_usage,
                            "usage_after": self.monitor.memory_usage(),
                        },
                    },
                )

    # ------------------------------------------------------------------
    # parameter optimization
    # ------------------------------------------------------------------
    def _optimize_parameters(self) -> None:
        if self.accuracy_threshold is None:
            return
        acc = self.monitor.current_accuracy()
        if acc >= self.accuracy_threshold:
            return
        for name, bounds in self.parameter_bounds.items():
            before = self.model_params[name]
            span = bounds[1] - bounds[0]
            delta = span * self.adjustment_factor
            after = min(bounds[1], before + delta)
            self.model_params[name] = after
            self.storage.store(
                "optimization",
                {
                    "metric": "accuracy",
                    "decision": {
                        "parameter": name,
                        "before": before,
                        "after": after,
                    },
                    "effect": {
                        "accuracy_before": acc,
                        "accuracy_after": self.monitor.current_accuracy(),
                    },
                },
            )

