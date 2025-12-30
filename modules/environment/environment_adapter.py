"""Environment-aware adaptation utilities.

The `EnvironmentAdapter` monitors local/system signals and the shared
`HardwareEnvironmentRegistry` to recommend (or apply) runtime adjustments:

- Concurrency throttling when CPU/memory pressure is sustained
- Suggesting different execution modes (local / ray / dask) when available
- Emitting a compact environment summary that can be injected into prompts

The module is intentionally lightweight and safe-by-default: it does not
perform any network calls and only applies changes when an explicit
`apply_callback` is provided.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Mapping, Optional

try:  # Optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

from .registry import get_hardware_registry, report_resource_signal


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _cpu_count() -> int:
    try:
        return int(os.cpu_count() or 1)
    except Exception:
        return 1


def _torch_gpu_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@dataclass(frozen=True)
class EnvironmentSnapshot:
    timestamp: float
    cpu_count: int
    cpu_percent: float | None
    memory_percent: float | None
    memory_total_gb: float | None
    gpu_available: bool
    registry_workers: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvironmentAdjustment:
    """Recommended runtime adjustments."""

    concurrency: int | None = None
    task_adapter_mode: str | None = None
    llm_model: str | None = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


ApplyCallback = Callable[[Dict[str, Any]], Any]


def choose_task_adapter_mode() -> str:
    """Select a task adapter mode based on deployment environment.

    Rules (safe fallback):
    - If Dask scheduler address is configured and Dask is importable -> "dask"
    - Else if Ray address is configured and Ray is importable -> "ray"
    - Else -> "local"
    """

    prefer = (os.getenv("TASK_ADAPTER_PREFERRED") or "").strip().lower()
    if prefer in {"local", "ray", "dask"}:
        return prefer

    dask_addr = os.getenv("DASK_SCHEDULER_ADDRESS") or os.getenv("DASK_ADDRESS")
    if dask_addr:
        try:
            import dask.distributed  # type: ignore  # noqa: F401

            return "dask"
        except Exception:
            pass

    ray_addr = os.getenv("RAY_ADDRESS") or os.getenv("RAY_HEAD_ADDRESS")
    if ray_addr:
        try:
            import ray  # type: ignore  # noqa: F401

            return "ray"
        except Exception:
            pass

    return "local"


def format_environment_prompt(snapshot: EnvironmentSnapshot) -> str:
    """Return a compact prompt snippet describing the runtime environment."""

    cpu = "unknown" if snapshot.cpu_percent is None else f"{snapshot.cpu_percent:.0f}%"
    mem = "unknown" if snapshot.memory_percent is None else f"{snapshot.memory_percent:.0f}%"
    mem_total = (
        "unknown"
        if snapshot.memory_total_gb is None
        else f"{snapshot.memory_total_gb:.1f}GB"
    )
    gpu = "available" if snapshot.gpu_available else "unavailable"
    return (
        "Runtime environment:\n"
        f"- cpu_cores: {snapshot.cpu_count}\n"
        f"- cpu_load: {cpu}\n"
        f"- memory_load: {mem}\n"
        f"- memory_total: {mem_total}\n"
        f"- gpu: {gpu}\n"
        "When planning, prefer resource-efficient steps when load is high.\n"
    )


class EnvironmentAdapter:
    """Monitor environment signals and recommend/apply safe adjustments."""

    def __init__(
        self,
        *,
        worker_id: str | None = None,
        event_bus: Any | None = None,
        apply_callback: ApplyCallback | None = None,
        interval_seconds: float = 10.0,
        cpu_high: float = 90.0,
        cpu_low: float = 60.0,
        mem_high: float = 85.0,
        mem_low: float = 65.0,
        sustain_samples: int = 3,
        min_concurrency: int = 1,
        max_concurrency: int | None = None,
        edge_mem_gb: float = 6.0,
    ) -> None:
        self.worker_id = worker_id or f"env-adapter:{os.getpid()}"
        self._bus = event_bus
        self._apply = apply_callback
        self._interval = max(0.2, float(interval_seconds))
        self._cpu_high = float(cpu_high)
        self._cpu_low = float(cpu_low)
        self._mem_high = float(mem_high)
        self._mem_low = float(mem_low)
        self._sustain = max(1, int(sustain_samples))
        self._min_conc = max(1, int(min_concurrency))
        self._max_conc = max_concurrency if max_concurrency is None else max(self._min_conc, int(max_concurrency))
        self._edge_mem_gb = max(0.5, float(edge_mem_gb))

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: Deque[Dict[str, float]] = deque(maxlen=self._sustain)
        self._last_adjustment: EnvironmentAdjustment | None = None

        # Register static capabilities (best-effort).
        try:
            get_hardware_registry().register(
                self.worker_id,
                {
                    "role": "environment_adapter",
                    "cpu_count": _cpu_count(),
                    "gpu_available": _torch_gpu_available(),
                },
                metadata={"pid": os.getpid()},
            )
        except Exception:
            pass

    @classmethod
    def from_env(
        cls,
        *,
        event_bus: Any | None = None,
        apply_callback: ApplyCallback | None = None,
    ) -> "EnvironmentAdapter | None":
        if not _parse_bool(os.getenv("ENVIRONMENT_ADAPTER_ENABLED"), default=False):
            return None
        interval = _safe_float(os.getenv("ENVIRONMENT_ADAPTER_INTERVAL_SECONDS"), 10.0)
        cpu_high = _safe_float(os.getenv("ENVIRONMENT_ADAPTER_CPU_HIGH"), 90.0)
        cpu_low = _safe_float(os.getenv("ENVIRONMENT_ADAPTER_CPU_LOW"), 60.0)
        mem_high = _safe_float(os.getenv("ENVIRONMENT_ADAPTER_MEM_HIGH"), 85.0)
        mem_low = _safe_float(os.getenv("ENVIRONMENT_ADAPTER_MEM_LOW"), 65.0)
        sustain = int(_safe_float(os.getenv("ENVIRONMENT_ADAPTER_SUSTAIN_SAMPLES"), 3))
        min_conc = int(_safe_float(os.getenv("ENVIRONMENT_ADAPTER_MIN_CONCURRENCY"), 1))
        max_conc_raw = os.getenv("ENVIRONMENT_ADAPTER_MAX_CONCURRENCY")
        max_conc = int(max_conc_raw) if max_conc_raw and max_conc_raw.isdigit() else None
        edge_mem_gb = _safe_float(os.getenv("ENVIRONMENT_ADAPTER_EDGE_MEM_GB"), 6.0)
        return cls(
            event_bus=event_bus,
            apply_callback=apply_callback,
            interval_seconds=interval,
            cpu_high=cpu_high,
            cpu_low=cpu_low,
            mem_high=mem_high,
            mem_low=mem_low,
            sustain_samples=sustain,
            min_concurrency=min_conc,
            max_concurrency=max_conc,
            edge_mem_gb=edge_mem_gb,
        )

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="environment-adapter", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float | None = 2.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    # ------------------------------------------------------------------
    def snapshot(self) -> EnvironmentSnapshot:
        now = time.time()
        cpu_cnt = _cpu_count()
        cpu_pct = None
        mem_pct = None
        mem_total = None
        if psutil is not None:
            try:
                cpu_pct = float(psutil.cpu_percent(interval=None))
            except Exception:
                cpu_pct = None
            try:
                vm = psutil.virtual_memory()
                mem_pct = float(getattr(vm, "percent", None))
                total_bytes = float(getattr(vm, "total", 0.0))
                if total_bytes > 0:
                    mem_total = total_bytes / (1024.0**3)
            except Exception:
                mem_pct = None
                mem_total = None
        gpu_avail = _torch_gpu_available()
        try:
            workers = get_hardware_registry().snapshot()
        except Exception:
            workers = {}
        return EnvironmentSnapshot(
            timestamp=now,
            cpu_count=cpu_cnt,
            cpu_percent=cpu_pct,
            memory_percent=mem_pct,
            memory_total_gb=mem_total,
            gpu_available=gpu_avail,
            registry_workers=workers if isinstance(workers, dict) else {},
        )

    def environment_prompt(self) -> str:
        return format_environment_prompt(self.snapshot())

    # ------------------------------------------------------------------
    def evaluate(self, snapshot: EnvironmentSnapshot) -> EnvironmentAdjustment:
        """Compute a recommended adjustment from a snapshot."""

        cpu_pct = snapshot.cpu_percent
        mem_pct = snapshot.memory_percent
        cpu_cnt = max(1, int(snapshot.cpu_count or 1))

        base_max = self._max_conc if self._max_conc is not None else max(self._min_conc, cpu_cnt)
        base_max = max(self._min_conc, base_max)

        is_edge = False
        if _parse_bool(os.getenv("EDGE_DEVICE"), default=False):
            is_edge = True
        elif snapshot.memory_total_gb is not None and snapshot.memory_total_gb <= self._edge_mem_gb:
            is_edge = True
        elif cpu_cnt <= 2 and (snapshot.memory_total_gb or 0.0) <= 4.0:
            is_edge = True

        # Track sustained pressure using a rolling window of samples.
        sample: Dict[str, float] = {}
        if isinstance(cpu_pct, (int, float)):
            sample["cpu_percent"] = float(cpu_pct)
        if isinstance(mem_pct, (int, float)):
            sample["memory_percent"] = float(mem_pct)
        with self._lock:
            if sample:
                self._samples.appendleft(sample)
            window = list(self._samples)

        avg_cpu = None
        avg_mem = None
        if window:
            cpu_vals = [s.get("cpu_percent") for s in window if "cpu_percent" in s]
            mem_vals = [s.get("memory_percent") for s in window if "memory_percent" in s]
            if cpu_vals:
                avg_cpu = sum(cpu_vals) / len(cpu_vals)
            if mem_vals:
                avg_mem = sum(mem_vals) / len(mem_vals)

        reason_parts = []
        desired_conc = base_max
        if avg_cpu is not None and avg_cpu >= self._cpu_high:
            desired_conc = max(self._min_conc, int(round(base_max * 0.5)))
            reason_parts.append(f"high_cpu(avg={avg_cpu:.1f}%)")
        elif avg_cpu is not None and avg_cpu <= self._cpu_low:
            reason_parts.append(f"cpu_ok(avg={avg_cpu:.1f}%)")

        if avg_mem is not None and avg_mem >= self._mem_high:
            desired_conc = max(self._min_conc, min(desired_conc, int(round(base_max * 0.5))))
            reason_parts.append(f"high_mem(avg={avg_mem:.1f}%)")
        elif avg_mem is not None and avg_mem <= self._mem_low:
            reason_parts.append(f"mem_ok(avg={avg_mem:.1f}%)")

        # Model recommendation: conservative by default on edge.
        llm_model = None
        if is_edge:
            llm_model = os.getenv("ENVIRONMENT_ADAPTER_MODEL_EDGE") or os.getenv("LLM_MODEL_EDGE")
            if llm_model:
                reason_parts.append("edge_model")
        else:
            if snapshot.gpu_available:
                llm_model = os.getenv("ENVIRONMENT_ADAPTER_MODEL_GPU") or os.getenv("LLM_MODEL_GPU")
                if llm_model:
                    reason_parts.append("gpu_model")
            if llm_model is None:
                llm_model = os.getenv("ENVIRONMENT_ADAPTER_MODEL_CPU") or os.getenv("LLM_MODEL_CPU")
                if llm_model:
                    reason_parts.append("cpu_model")

        # Execution mode recommendation (startup-time): auto-detect cluster if configured.
        mode = choose_task_adapter_mode()
        if mode != "local":
            reason_parts.append(f"prefer_{mode}")

        meta = {
            "avg_cpu_percent": avg_cpu,
            "avg_memory_percent": avg_mem,
            "edge": is_edge,
            "gpu_available": snapshot.gpu_available,
            "cpu_count": cpu_cnt,
            "base_max_concurrency": base_max,
        }
        return EnvironmentAdjustment(
            concurrency=int(desired_conc),
            task_adapter_mode=mode,
            llm_model=llm_model,
            reason=",".join(reason_parts) if reason_parts else "no_signal",
            metadata=meta,
        )

    def run_once(self) -> EnvironmentAdjustment:
        snap = self.snapshot()

        # Publish resource signals to the shared registry.
        signal: Dict[str, float] = {}
        if snap.cpu_percent is not None:
            signal["cpu_percent"] = float(snap.cpu_percent)
            signal["cpu_utilization"] = float(snap.cpu_percent) / 100.0
        if snap.memory_percent is not None:
            signal["memory_percent"] = float(snap.memory_percent)
            signal["memory_utilization"] = float(snap.memory_percent) / 100.0
        if signal:
            try:
                report_resource_signal(
                    self.worker_id,
                    signal,
                    metadata={"source": "environment_adapter"},
                    event_bus=self._bus,
                )
            except Exception:
                pass

        adjustment = self.evaluate(snap)
        self._maybe_apply(adjustment, snap)
        return adjustment

    # ------------------------------------------------------------------
    def _maybe_apply(self, adjustment: EnvironmentAdjustment, snapshot: EnvironmentSnapshot) -> None:
        last = self._last_adjustment
        if last == adjustment:
            return
        with self._lock:
            self._last_adjustment = adjustment

        payload = {
            "worker_id": self.worker_id,
            "timestamp": snapshot.timestamp,
            "adjustment": {
                "concurrency": adjustment.concurrency,
                "task_adapter_mode": adjustment.task_adapter_mode,
                "llm_model": adjustment.llm_model,
                "reason": adjustment.reason,
                "metadata": dict(adjustment.metadata),
            },
        }

        if self._bus is not None:
            try:
                self._bus.publish("environment.adjustment", payload)
            except Exception:
                pass

        if self._apply is None:
            return

        apply_dict: Dict[str, Any] = {}
        if adjustment.concurrency is not None:
            apply_dict["concurrency"] = int(adjustment.concurrency)
        if adjustment.llm_model:
            apply_dict["llm_model"] = str(adjustment.llm_model)
        if adjustment.task_adapter_mode:
            apply_dict["task_adapter_mode"] = str(adjustment.task_adapter_mode)
        apply_dict["reason"] = adjustment.reason
        apply_dict["metadata"] = dict(adjustment.metadata)

        try:
            self._apply(apply_dict)
        except Exception:
            pass

    def _run(self) -> None:  # pragma: no cover - background thread
        while not self._stop.wait(self._interval):
            try:
                self.run_once()
            except Exception:
                continue


__all__ = [
    "EnvironmentAdapter",
    "EnvironmentSnapshot",
    "EnvironmentAdjustment",
    "choose_task_adapter_mode",
    "format_environment_prompt",
]

