"""Resource management helpers for edge devices."""
from __future__ import annotations

try:  # Optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore


class EdgeResourceManager:
    """Monitor compute resources on edge hardware."""

    def cpu_usage(self) -> float:
        """Return system-wide CPU utilisation percentage."""
        if psutil is None:
            return 0.0
        cpu_percent = getattr(psutil, "cpu_percent", None)
        if not callable(cpu_percent):
            return 0.0
        try:
            return float(cpu_percent(interval=None))
        except Exception:
            return 0.0

    def memory_usage(self) -> float:
        """Return system memory utilisation percentage."""
        if psutil is None:
            return 0.0
        virtual_memory = getattr(psutil, "virtual_memory", None)
        if not callable(virtual_memory):
            return 0.0
        try:
            return float(getattr(virtual_memory(), "percent", 0.0))
        except Exception:
            return 0.0

    def gpu_available(self) -> bool:
        """Return True if a CUDA-capable GPU is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False
