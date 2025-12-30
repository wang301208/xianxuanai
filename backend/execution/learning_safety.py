from __future__ import annotations

"""Safety guardrails for continual learning.

The guard monitors a lightweight score derived from existing runtime metrics and
triggers early-stop or rollback when learning appears to regress performance.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

try:  # Optional for typing; guard can run without an event bus.
    from events import EventBus  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    EventBus = None  # type: ignore

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return int(default)
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    return number if _is_finite(number) else None


def _mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / float(len(values))


def _metric_from_monitor(monitor: Any, key: str, window: int) -> Optional[float]:
    if monitor is None:
        return None
    snapshots = getattr(monitor, "snapshots", None)
    if not isinstance(snapshots, list) or not snapshots:
        return None

    window = int(max(1, window))
    recent = snapshots[-window:]
    values: list[float] = []
    for _, payload in recent:
        if not isinstance(payload, Mapping):
            continue
        if key not in payload:
            continue
        number = _safe_float(payload.get(key))
        if number is None:
            continue
        values.append(number)
    return _mean(values)


@dataclass
class _BestSnapshot:
    score: float
    captured_at: float
    predictive_state: Any | None
    imitation_state: Any | None


class LearningSafetyGuard:
    """Early-stop and rollback controller for background learning cycles."""

    ACTION_NONE = 0.0
    ACTION_IMPROVED = 1.0
    ACTION_PAUSED = 2.0
    ACTION_ROLLBACK = 3.0
    ACTION_INIT = 4.0

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        patience: int | None = None,
        min_delta: float | None = None,
        rollback_delta: float | None = None,
        pause_seconds: float | None = None,
        monitor_window: int | None = None,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self._enabled = _env_bool("LEARNING_GUARD_ENABLED", True) if enabled is None else bool(enabled)
        self._patience = _env_int("LEARNING_GUARD_PATIENCE", 3) if patience is None else int(patience)
        self._min_delta = _env_float("LEARNING_GUARD_MIN_DELTA", 1e-3) if min_delta is None else float(min_delta)
        self._rollback_delta = (
            _env_float("LEARNING_GUARD_ROLLBACK_DELTA", 0.05) if rollback_delta is None else float(rollback_delta)
        )
        self._pause_seconds = (
            _env_float("LEARNING_GUARD_PAUSE_SECS", 300.0) if pause_seconds is None else float(pause_seconds)
        )
        self._monitor_window = (
            _env_int("LEARNING_GUARD_MONITOR_WINDOW", 50) if monitor_window is None else int(monitor_window)
        )

        self._w_success = _env_float("LEARNING_GUARD_W_SUCCESS", 1.0)
        self._w_reward = _env_float("LEARNING_GUARD_W_REWARD", 0.05)
        self._w_cross = _env_float("LEARNING_GUARD_W_CROSS", 1.0)
        self._w_pred_err = _env_float("LEARNING_GUARD_W_PRED_ERROR", 0.1)
        self._w_imitation_acc = _env_float("LEARNING_GUARD_W_IMITATION_ACC", 0.2)
        self._w_imitation_loss = _env_float("LEARNING_GUARD_W_IMITATION_LOSS", 0.1)

        self._logger = logger_ or logger

        self._best: _BestSnapshot | None = None
        self._no_improve: int = 0
        self._last_score: float | None = None

    def evaluate(
        self,
        *,
        stats: Mapping[str, Any],
        performance_monitor: Any,
        predictive_model: Any,
        imitation_policy: Any,
        learning_manager: Any | None = None,
        event_bus: Optional[EventBus] = None,
        now: float | None = None,
    ) -> Dict[str, float]:
        """Update guard state and trigger rollback/pause when needed."""

        if not self._enabled:
            return {}

        now_ts = time.time() if now is None else float(now)
        score, used = self._compute_score(stats, performance_monitor)
        if score is None:
            return {}

        action = self.ACTION_NONE
        if self._best is None:
            self._best = _BestSnapshot(
                score=score,
                captured_at=now_ts,
                predictive_state=self._snapshot_model(predictive_model),
                imitation_state=self._snapshot_model(imitation_policy),
            )
            self._no_improve = 0
            action = self.ACTION_INIT
        else:
            improved = score > float(self._best.score) + float(self._min_delta)
            if improved:
                self._best = _BestSnapshot(
                    score=score,
                    captured_at=now_ts,
                    predictive_state=self._snapshot_model(predictive_model),
                    imitation_state=self._snapshot_model(imitation_policy),
                )
                self._no_improve = 0
                action = self.ACTION_IMPROVED
            else:
                self._no_improve += 1

            regressed = score < float(self._best.score) - float(self._rollback_delta)
            if regressed and self._rollback_delta > 0 and self._best is not None:
                restored = self._restore_best(predictive_model, imitation_policy)
                if restored:
                    action = self.ACTION_ROLLBACK
                    self._no_improve = 0
                    self._pause_learning(learning_manager, reason="rollback")
                    self._publish_event(
                        event_bus,
                        action="rollback",
                        now=now_ts,
                        score=score,
                        best_score=float(self._best.score),
                        no_improve=self._no_improve,
                        metrics=used,
                    )
            elif self._patience > 0 and self._no_improve >= self._patience:
                action = self.ACTION_PAUSED
                self._no_improve = 0
                self._pause_learning(learning_manager, reason="early_stop")
                self._publish_event(
                    event_bus,
                    action="paused",
                    now=now_ts,
                    score=score,
                    best_score=float(self._best.score),
                    no_improve=self._no_improve,
                    metrics=used,
                )

        self._last_score = score
        best_score = float(self._best.score) if self._best is not None else score
        return {
            "learning_guard_score": float(score),
            "learning_guard_best_score": float(best_score),
            "learning_guard_no_improve": float(self._no_improve),
            "learning_guard_action": float(action),
            "learning_guard_metrics_used": float(len(used)),
        }

    # ------------------------------------------------------------------
    def _compute_score(
        self, stats: Mapping[str, Any], performance_monitor: Any
    ) -> Tuple[Optional[float], Dict[str, float]]:
        used: Dict[str, float] = {}

        success = _metric_from_monitor(performance_monitor, "decision_success_rate", self._monitor_window)
        reward = _metric_from_monitor(performance_monitor, "decision_reward_avg", self._monitor_window)
        cross = _safe_float(stats.get("cross_domain_success"))
        if cross is None:
            cross = _metric_from_monitor(performance_monitor, "cross_domain_success", self._monitor_window)
        pred_error = _safe_float(stats.get("online_prediction_error"))
        imitation_acc = _safe_float(stats.get("imitation_accuracy"))
        imitation_loss = _safe_float(stats.get("imitation_loss"))

        score = 0.0
        if success is not None and self._w_success != 0.0:
            used["decision_success_rate"] = float(success)
            score += float(self._w_success) * float(success)
        if reward is not None and self._w_reward != 0.0:
            used["decision_reward_avg"] = float(reward)
            score += float(self._w_reward) * float(reward)
        if cross is not None and self._w_cross != 0.0:
            used["cross_domain_success"] = float(cross)
            score += float(self._w_cross) * float(cross)
        if pred_error is not None and self._w_pred_err != 0.0:
            used["online_prediction_error"] = float(pred_error)
            score -= float(self._w_pred_err) * float(pred_error)
        if imitation_acc is not None and self._w_imitation_acc != 0.0:
            used["imitation_accuracy"] = float(imitation_acc)
            score += float(self._w_imitation_acc) * float(imitation_acc)
        if imitation_loss is not None and self._w_imitation_loss != 0.0:
            used["imitation_loss"] = float(imitation_loss)
            score -= float(self._w_imitation_loss) * float(imitation_loss)

        if not used:
            return None, {}
        if not _is_finite(float(score)):
            return None, {}
        return float(score), used

    def _snapshot_model(self, model: Any) -> Any | None:
        if model is None:
            return None
        getter = getattr(model, "get_state", None)
        if callable(getter):
            try:
                return getter()
            except Exception:
                self._logger.debug("Failed to snapshot model state", exc_info=True)
                return None
        return None

    def _restore_model(self, model: Any, state: Any) -> bool:
        if model is None or state is None:
            return False
        setter = getattr(model, "set_state", None)
        if callable(setter):
            try:
                setter(state)
                return True
            except Exception:
                self._logger.debug("Failed to restore model state", exc_info=True)
                return False
        return False

    def _restore_best(self, predictive_model: Any, imitation_policy: Any) -> bool:
        if self._best is None:
            return False
        restored_any = False
        restored_any |= self._restore_model(predictive_model, self._best.predictive_state)
        restored_any |= self._restore_model(imitation_policy, self._best.imitation_state)
        return restored_any

    def _pause_learning(self, learning_manager: Any | None, *, reason: str) -> None:
        if learning_manager is None or self._pause_seconds <= 0:
            return
        pause = getattr(learning_manager, "pause", None)
        if not callable(pause):
            return
        try:
            pause(float(self._pause_seconds), reason=reason)
        except Exception:
            return

    def _publish_event(
        self,
        event_bus: Optional[EventBus],
        *,
        action: str,
        now: float,
        score: float,
        best_score: float,
        no_improve: int,
        metrics: Mapping[str, float],
    ) -> None:
        if event_bus is None:
            return
        try:
            event_bus.publish(
                "learning.guard",
                {
                    "time": float(now),
                    "action": str(action),
                    "score": float(score),
                    "best_score": float(best_score),
                    "no_improve": int(no_improve),
                    "metrics": dict(metrics),
                },
            )
        except Exception:
            return


__all__ = ["LearningSafetyGuard"]

