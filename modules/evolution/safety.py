"""Safety utilities for self-evolution loops.

This module provides optional guardrails for architecture/self-improvement
updates:
- Performance gating (reject candidates that fall below a historical baseline)
- Manual-review queue for large changes
- Pluggable sandbox/regression runners (e.g. pytest suites)

All features are opt-in and designed to be dependency-light.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on", "enabled"}


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def architecture_delta_l1(old: Mapping[str, float], new: Mapping[str, float]) -> float:
    keys = set(old.keys()) | set(new.keys())
    total = 0.0
    for key in keys:
        try:
            a = float(old.get(key, 0.0))
            b = float(new.get(key, 0.0))
        except Exception:
            continue
        total += abs(b - a)
    return float(total)


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    reason: str
    baseline_mean: Optional[float] = None
    baseline_window: Optional[int] = None
    candidate_performance: Optional[float] = None
    delta_l1: float = 0.0
    requires_review: bool = False
    review_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvolutionSafetyConfig:
    """Configuration for evolution safety gates."""

    enabled: bool = False

    # Performance gating: candidate must satisfy at least one of these (when set).
    history_window: int = 5
    min_relative_to_mean: float = 0.0  # e.g. 0.95 => >= 95% of rolling mean
    max_drop_from_mean: float = 0.0  # absolute drop allowed vs rolling mean
    min_performance: Optional[float] = None

    # Manual review gate for large deltas.
    manual_review_enabled: bool = False
    manual_review_delta_l1: float = 0.0
    max_pending_reviews: int = 1000

    # Sandbox/regression runner.
    sandbox_enabled: bool = False
    sandbox_timeout_s: Optional[float] = None
    sandbox_pytest_paths: Tuple[str, ...] = ()
    sandbox_pytest_args: Tuple[str, ...] = ()

    @classmethod
    def from_sources(cls, config: Optional[Mapping[str, Any]] = None) -> "EvolutionSafetyConfig":
        """Build config from an optional mapping and environment variables."""

        cfg = dict(config or {})

        enabled = _parse_bool(cfg.get("enabled", os.environ.get("BSS_EVOLUTION_SAFETY_ENABLED", "")))
        history_window = max(
            1,
            _safe_int(cfg.get("history_window", os.environ.get("BSS_EVOLUTION_SAFETY_WINDOW", 5)), 5),
        )
        min_relative = _safe_float(cfg.get("min_relative_to_mean", os.environ.get("BSS_EVOLUTION_SAFETY_MIN_RELATIVE", 0.0)), 0.0)
        max_drop = _safe_float(cfg.get("max_drop_from_mean", os.environ.get("BSS_EVOLUTION_SAFETY_MAX_DROP", 0.0)), 0.0)

        min_perf_raw = cfg.get("min_performance", os.environ.get("BSS_EVOLUTION_SAFETY_MIN_PERFORMANCE", ""))
        min_perf = None
        if str(min_perf_raw).strip() != "":
            min_perf = _safe_float(min_perf_raw, 0.0)

        manual_enabled = _parse_bool(cfg.get("manual_review_enabled", os.environ.get("BSS_EVOLUTION_MANUAL_REVIEW", "")))
        manual_delta = _safe_float(cfg.get("manual_review_delta_l1", os.environ.get("BSS_EVOLUTION_MANUAL_REVIEW_DELTA_L1", 0.0)), 0.0)
        max_pending = max(
            1,
            _safe_int(cfg.get("max_pending_reviews", os.environ.get("BSS_EVOLUTION_MANUAL_REVIEW_MAX_PENDING", 1000)), 1000),
        )

        sandbox_cfg = cfg.get("sandbox") if isinstance(cfg.get("sandbox"), Mapping) else {}
        sandbox_enabled = _parse_bool(
            cfg.get("sandbox_enabled", sandbox_cfg.get("enabled", os.environ.get("BSS_EVOLUTION_SANDBOX_ENABLED", "")))
        )
        timeout_raw = cfg.get("sandbox_timeout_s", sandbox_cfg.get("timeout_s", os.environ.get("BSS_EVOLUTION_SANDBOX_TIMEOUT_S", "")))
        timeout = None
        if str(timeout_raw).strip() != "":
            timeout = _safe_float(timeout_raw, 0.0)

        paths_val = cfg.get(
            "sandbox_pytest_paths",
            sandbox_cfg.get("pytest_paths", os.environ.get("BSS_EVOLUTION_SANDBOX_PYTEST_PATHS", "")),
        )
        pytest_paths: Tuple[str, ...] = ()
        if isinstance(paths_val, str):
            parts = [p.strip() for p in paths_val.replace(";", ",").split(",") if p.strip()]
            pytest_paths = tuple(parts)
        elif isinstance(paths_val, (list, tuple)):
            pytest_paths = tuple(str(p).strip() for p in paths_val if str(p).strip())

        args_val = cfg.get(
            "sandbox_pytest_args",
            sandbox_cfg.get("pytest_args", os.environ.get("BSS_EVOLUTION_SANDBOX_PYTEST_ARGS", "")),
        )
        pytest_args: Tuple[str, ...] = ()
        if isinstance(args_val, str):
            parts = [p.strip() for p in args_val.replace(";", ",").split(",") if p.strip()]
            pytest_args = tuple(parts)
        elif isinstance(args_val, (list, tuple)):
            pytest_args = tuple(str(p).strip() for p in args_val if str(p).strip())

        return cls(
            enabled=bool(enabled),
            history_window=history_window,
            min_relative_to_mean=max(0.0, float(min_relative)),
            max_drop_from_mean=max(0.0, float(max_drop)),
            min_performance=min_perf,
            manual_review_enabled=bool(manual_enabled),
            manual_review_delta_l1=max(0.0, float(manual_delta)),
            max_pending_reviews=max_pending,
            sandbox_enabled=bool(sandbox_enabled),
            sandbox_timeout_s=timeout,
            sandbox_pytest_paths=pytest_paths,
            sandbox_pytest_args=pytest_args,
        )


@dataclass(frozen=True)
class PendingArchitectureUpdate:
    id: str
    created_at: float
    architecture: Dict[str, float]
    performance: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    reason: str = "manual_review"
    delta_l1: float = 0.0


class ArchitectureApprovalQueue:
    """In-memory queue for manual review of proposed architecture updates."""

    def __init__(self, *, max_pending: int = 1000) -> None:
        self._lock = threading.Lock()
        self._max_pending = max(1, int(max_pending))
        self._pending: Dict[str, PendingArchitectureUpdate] = {}

    def request(
        self,
        architecture: Mapping[str, float],
        *,
        performance: float,
        metrics: Optional[Mapping[str, Any]] = None,
        reason: str = "manual_review",
        delta_l1: float = 0.0,
    ) -> PendingArchitectureUpdate:
        now = time.time()
        update_id = uuid.uuid4().hex
        update = PendingArchitectureUpdate(
            id=update_id,
            created_at=now,
            architecture=dict(architecture),
            performance=float(performance),
            metrics=dict(metrics or {}),
            reason=str(reason or "manual_review"),
            delta_l1=float(delta_l1),
        )
        with self._lock:
            if len(self._pending) >= self._max_pending:
                # Evict oldest pending request.
                oldest = min(self._pending.values(), key=lambda item: item.created_at)
                self._pending.pop(oldest.id, None)
            self._pending[update.id] = update
        return update

    def get(self, update_id: str) -> Optional[PendingArchitectureUpdate]:
        with self._lock:
            return self._pending.get(str(update_id))

    def pop(self, update_id: str) -> Optional[PendingArchitectureUpdate]:
        with self._lock:
            return self._pending.pop(str(update_id), None)

    def list(self, *, limit: int = 50) -> List[PendingArchitectureUpdate]:
        limit = max(0, int(limit))
        with self._lock:
            items = list(self._pending.values())
        items.sort(key=lambda item: item.created_at, reverse=True)
        return items[:limit] if limit else items


class PytestSandboxRunner:
    """Run a pytest regression suite to validate a candidate update."""

    def __init__(
        self,
        paths: Sequence[str] | None = None,
        *,
        timeout_s: float | None = None,
        extra_args: Sequence[str] | None = None,
        cwd: str | None = None,
    ) -> None:
        self.paths = list(paths or [])
        self.timeout_s = float(timeout_s) if timeout_s is not None else None
        self.extra_args = list(extra_args or [])
        self.cwd = cwd

    def __call__(self) -> bool:
        if not self.paths:
            return True
        cmd = [sys.executable, "-m", "pytest", *self.extra_args, *self.paths]
        try:
            result = subprocess.run(cmd, check=False, timeout=self.timeout_s, cwd=self.cwd)
        except Exception:
            return False
        return result.returncode == 0


def evaluate_candidate(
    *,
    config: EvolutionSafetyConfig,
    history_performances: Sequence[float],
    current_architecture: Mapping[str, float],
    candidate_architecture: Mapping[str, float],
    candidate_performance: float,
    approval_queue: Optional[ArchitectureApprovalQueue] = None,
    extra_details: Optional[Mapping[str, Any]] = None,
) -> SafetyDecision:
    """Evaluate a candidate update against safety gates."""

    details = dict(extra_details or {})

    delta = architecture_delta_l1(current_architecture, candidate_architecture)
    details["delta_l1"] = float(delta)

    baseline_mean: Optional[float] = None
    window = max(1, int(config.history_window))
    if history_performances:
        windowed = list(history_performances)[-window:]
        if windowed:
            baseline_mean = sum(windowed) / len(windowed)
            details["baseline_window"] = len(windowed)
            details["baseline_mean"] = float(baseline_mean)

    # Manual review gate for large changes.
    if config.manual_review_enabled and config.manual_review_delta_l1 > 0 and delta >= config.manual_review_delta_l1:
        queued = None
        if approval_queue is not None:
            queued = approval_queue.request(
                candidate_architecture,
                performance=candidate_performance,
                metrics=details,
                reason="manual_review_required",
                delta_l1=delta,
            )
        return SafetyDecision(
            allowed=False,
            reason="manual_review_required",
            baseline_mean=baseline_mean,
            baseline_window=window,
            candidate_performance=float(candidate_performance),
            delta_l1=float(delta),
            requires_review=True,
            review_id=queued.id if queued is not None else None,
            details=details,
        )

    # Performance floor gate.
    if config.min_performance is not None and candidate_performance < float(config.min_performance):
        return SafetyDecision(
            allowed=False,
            reason="below_min_performance",
            baseline_mean=baseline_mean,
            baseline_window=window,
            candidate_performance=float(candidate_performance),
            delta_l1=float(delta),
            details=details,
        )

    if baseline_mean is not None:
        min_relative = float(config.min_relative_to_mean)
        if min_relative > 0.0 and candidate_performance < baseline_mean * min_relative:
            details["min_relative_to_mean"] = min_relative
            return SafetyDecision(
                allowed=False,
                reason="below_relative_baseline",
                baseline_mean=float(baseline_mean),
                baseline_window=window,
                candidate_performance=float(candidate_performance),
                delta_l1=float(delta),
                details=details,
            )
        max_drop = float(config.max_drop_from_mean)
        if max_drop > 0.0 and candidate_performance < baseline_mean - max_drop:
            details["max_drop_from_mean"] = max_drop
            return SafetyDecision(
                allowed=False,
                reason="below_absolute_baseline",
                baseline_mean=float(baseline_mean),
                baseline_window=window,
                candidate_performance=float(candidate_performance),
                delta_l1=float(delta),
                details=details,
            )

    return SafetyDecision(
        allowed=True,
        reason="accepted",
        baseline_mean=baseline_mean,
        baseline_window=window,
        candidate_performance=float(candidate_performance),
        delta_l1=float(delta),
        details=details,
    )


__all__ = [
    "EvolutionSafetyConfig",
    "SafetyDecision",
    "PendingArchitectureUpdate",
    "ArchitectureApprovalQueue",
    "PytestSandboxRunner",
    "architecture_delta_l1",
    "evaluate_candidate",
]

