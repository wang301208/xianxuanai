"""Resource management helpers for edge devices."""
from __future__ import annotations

import psutil


class EdgeResourceManager:
    """Monitor compute resources on edge hardware."""

    def cpu_usage(self) -> float:
        """Return system-wide CPU utilisation percentage."""
        return psutil.cpu_percent(interval=None)

    def memory_usage(self) -> float:
        """Return system memory utilisation percentage."""
        return psutil.virtual_memory().percent

    def gpu_available(self) -> bool:
        """Return True if a CUDA-capable GPU is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False
