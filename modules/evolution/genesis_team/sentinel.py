"""Sentinel agent monitors for anomalies."""

from __future__ import annotations

import os
import platform
import shutil
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .. import Agent


@dataclass(frozen=True)
class SentinelReport:
    status: str
    signals: Dict[str, Any]
    warnings: List[str]


class Sentinel(Agent):
    """Monitors system state for anomalies."""

    def __init__(
        self,
        *,
        disk_warning_free_gb: float = 1.0,
        cpu_warning_percent: float = 95.0,
        memory_warning_percent: float = 95.0,
    ) -> None:
        self.disk_warning_free_gb = disk_warning_free_gb
        self.cpu_warning_percent = cpu_warning_percent
        self.memory_warning_percent = memory_warning_percent

    def perform(self) -> str:
        report = self.collect()
        warnings = "; ".join(report.warnings) if report.warnings else "none"
        return (
            f"Sentinel status={report.status} "
            f"cpu={report.signals.get('cpu_percent')}% "
            f"mem={report.signals.get('memory_percent')}% "
            f"disk_free_gb={report.signals.get('disk_free_gb')} "
            f"warnings={warnings}"
        )

    def collect(self) -> SentinelReport:
        warnings: List[str] = []
        signals: Dict[str, Any] = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "timestamp": time.time(),
            "cpu_count": os.cpu_count() or 1,
        }

        signals.update(self._collect_resource_usage())
        disk_free = float(signals.get("disk_free_gb") or 0.0)
        if disk_free and disk_free < self.disk_warning_free_gb:
            warnings.append(f"low_disk_free_gb<{self.disk_warning_free_gb}")
        cpu_percent = signals.get("cpu_percent")
        if isinstance(cpu_percent, (int, float)) and cpu_percent >= self.cpu_warning_percent:
            warnings.append(f"high_cpu>={self.cpu_warning_percent}")
        mem_percent = signals.get("memory_percent")
        if isinstance(mem_percent, (int, float)) and mem_percent >= self.memory_warning_percent:
            warnings.append(f"high_memory>={self.memory_warning_percent}")

        status = "warning" if warnings else "ok"
        return SentinelReport(status=status, signals=signals, warnings=warnings)

    def _collect_resource_usage(self) -> Dict[str, Any]:
        usage: Dict[str, Any] = {}
        try:
            total, used, free = shutil.disk_usage(os.getcwd())
            usage["disk_free_gb"] = round(free / (1024**3), 3)
            usage["disk_total_gb"] = round(total / (1024**3), 3)
        except Exception:
            usage["disk_free_gb"] = None

        psutil = _maybe_import_psutil()
        if psutil is None:
            usage["cpu_percent"] = None
            usage["memory_percent"] = None
            return usage

        try:
            usage["cpu_percent"] = float(psutil.cpu_percent(interval=0.1))
        except Exception:
            usage["cpu_percent"] = None
        try:
            usage["memory_percent"] = float(psutil.virtual_memory().percent)
        except Exception:
            usage["memory_percent"] = None
        return usage


def _maybe_import_psutil():
    try:  # pragma: no cover - optional dependency
        import psutil  # type: ignore
    except Exception:
        return None
    return psutil


__all__ = ["Sentinel", "SentinelReport"]
