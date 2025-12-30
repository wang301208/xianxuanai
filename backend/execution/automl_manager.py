from __future__ import annotations

"""Background AutoML hyper-parameter search (Optuna-first, fallback built-in).

This module is an *opt-in* low-priority background process:
- it listens for `automl.request` events (typically emitted by SelfImprovementManager)
- generates a candidate hyper-parameter configuration (a "suggestion")
- publishes `automl.suggestion` for SelfImprovementManager to apply/evaluate
- receives `automl.feedback` (objective value) and updates the optimiser state

Optuna integration is optional. When Optuna is not installed, the manager falls
back to a deterministic random sampler.
"""

import logging
import math
import os
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from events import EventBus

from .task_manager import TaskManager, TaskPriority

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


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str  # float|int|categorical
    low: float | None = None
    high: float | None = None
    log: bool = False
    step: float | None = None
    choices: Sequence[Any] | None = None


class AutoMLManager:
    """Generate and track AutoML hyper-parameter suggestions."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        task_manager: TaskManager,
        enabled: bool | None = None,
        backend: str | None = None,
        cooldown_secs: float | None = None,
        seed: int | None = None,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        self._bus = event_bus
        self._task_manager = task_manager
        self._logger = logger_ or logger

        self._enabled = _env_bool("AUTOML_ENABLED", False) if enabled is None else bool(enabled)
        self._backend = str(os.getenv("AUTOML_BACKEND", "optuna") if backend is None else backend).strip().lower()
        self._cooldown_secs = _env_float("AUTOML_COOLDOWN_SECS", 600.0) if cooldown_secs is None else float(cooldown_secs)
        self._seed = _env_int("AUTOML_SEED", 0) if seed is None else int(seed)
        self._rng = random.Random(self._seed)

        self._last_request_ts: Dict[str, float] = {}
        self._inflight_by_metric: Dict[str, str] = {}
        self._inflight: Dict[str, Dict[str, Any]] = {}
        self._studies: Dict[str, Any] = {}

        self._subscriptions: list[Callable[[], None]] = [
            self._bus.subscribe("automl.request", self._on_request),
            self._bus.subscribe("automl.feedback", self._on_feedback),
        ]

    def close(self) -> None:
        subs = list(self._subscriptions)
        self._subscriptions.clear()
        for cancel in subs:
            try:
                cancel()
            except Exception:
                continue

    # ------------------------------------------------------------------ event handlers
    async def _on_request(self, event: Dict[str, Any]) -> None:
        if not self._enabled or not isinstance(event, Mapping):
            return
        metric = str(event.get("metric") or "").strip() or "decision_success_rate"
        now = float(event.get("time", time.time()) or time.time())
        last = float(self._last_request_ts.get(metric, 0.0) or 0.0)
        if self._cooldown_secs > 0 and (now - last) < self._cooldown_secs:
            return
        if metric in self._inflight_by_metric:
            return

        request = dict(event)
        self._last_request_ts[metric] = now

        try:
            self._task_manager.submit(
                self._generate_and_publish,
                metric,
                request,
                priority=TaskPriority.LOW,
                category="automl",
                deadline=now + 300.0,
                metadata={"metric": metric, "source": "automl.request"},
            )
        except Exception:
            # Fall back to inline generation (best-effort).
            try:
                self._generate_and_publish(metric, request)
            except Exception:
                self._logger.debug("AutoML suggestion generation failed", exc_info=True)

    async def _on_feedback(self, event: Dict[str, Any]) -> None:
        if not self._enabled or not isinstance(event, Mapping):
            return
        suggestion_id = str(event.get("suggestion_id") or "").strip()
        metric = str(event.get("metric") or "").strip()
        if not suggestion_id or not metric:
            return
        value = event.get("objective_value")
        try:
            score = float(value)
        except Exception:
            return

        inflight = self._inflight.pop(suggestion_id, None)
        if inflight is None:
            return
        if self._inflight_by_metric.get(metric) == suggestion_id:
            self._inflight_by_metric.pop(metric, None)

        if inflight.get("backend") == "optuna" and inflight.get("trial") is not None:
            self._tell_optuna(metric, inflight["trial"], score)
        # For fallback backend we simply record; future versions can build a surrogate.
        try:
            self._bus.publish(
                "automl.trial_completed",
                {"time": time.time(), "metric": metric, "suggestion_id": suggestion_id, "objective_value": score},
            )
        except Exception:
            pass

    # ------------------------------------------------------------------ generation
    def _generate_and_publish(self, metric: str, request: Mapping[str, Any]) -> None:
        if not self._enabled:
            return
        metric = str(metric or "").strip() or "decision_success_rate"
        if metric in self._inflight_by_metric:
            return
        suggestion_id = uuid.uuid4().hex

        spec = request.get("space")
        space = self._parse_space(spec) if spec is not None else self._default_space(metric)
        params: Dict[str, Any] = {}
        backend_used = "random"
        trial = None

        if self._backend in {"optuna", "optuna_tpe"}:
            trial, params = self._suggest_optuna(metric, space)
            if params:
                backend_used = "optuna"

        if not params:
            params = self._suggest_random(space)
            backend_used = "random"

        self._inflight[suggestion_id] = {
            "metric": metric,
            "backend": backend_used,
            "trial": trial,
            "params": dict(params),
            "requested": dict(request),
            "time": time.time(),
        }
        self._inflight_by_metric[metric] = suggestion_id

        payload: Dict[str, Any] = {
            "time": time.time(),
            "suggestion_id": suggestion_id,
            "metric": metric,
            "params": dict(params),
            "source": f"automl:{backend_used}",
        }
        for key in ("eval_window", "min_improvement", "direction", "target"):
            if key in request:
                payload[key] = request.get(key)
        try:
            self._bus.publish("automl.suggestion", payload)
        except Exception:
            pass

    # ------------------------------------------------------------------ optuna integration (optional)
    def _suggest_optuna(self, metric: str, space: Sequence[ParamSpec]) -> tuple[Any | None, Dict[str, Any]]:
        try:
            import optuna  # type: ignore
        except Exception:
            return None, {}

        study = self._studies.get(metric)
        if study is None:
            try:
                sampler = optuna.samplers.TPESampler(seed=self._seed)
            except Exception:
                sampler = None
            study = optuna.create_study(direction="maximize", sampler=sampler)
            self._studies[metric] = study

        trial = study.ask()
        params: Dict[str, Any] = {}
        for spec in space:
            kind = str(spec.kind).strip().lower()
            name = str(spec.name)
            if kind == "categorical":
                choices = list(spec.choices or ())
                if not choices:
                    continue
                params[name] = trial.suggest_categorical(name, choices)
            elif kind == "int":
                if spec.low is None or spec.high is None:
                    continue
                params[name] = trial.suggest_int(name, int(spec.low), int(spec.high))
            else:
                if spec.low is None or spec.high is None:
                    continue
                step = float(spec.step) if spec.step is not None else None
                params[name] = trial.suggest_float(name, float(spec.low), float(spec.high), log=bool(spec.log), step=step)
        return trial, params

    def _tell_optuna(self, metric: str, trial: Any, value: float) -> None:
        try:
            import optuna  # type: ignore
        except Exception:
            return
        study = self._studies.get(metric)
        if study is None:
            return
        try:
            study.tell(trial, float(value))
        except Exception:
            # Optuna may reject tell for already-finished trials; ignore.
            return

    # ------------------------------------------------------------------ fallback samplers
    def _suggest_random(self, space: Sequence[ParamSpec]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for spec in space:
            kind = str(spec.kind).strip().lower()
            if kind == "categorical":
                choices = list(spec.choices or ())
                if not choices:
                    continue
                params[spec.name] = self._rng.choice(choices)
                continue
            if spec.low is None or spec.high is None:
                continue
            low = float(spec.low)
            high = float(spec.high)
            if kind == "int":
                params[spec.name] = int(self._rng.randint(int(low), int(high)))
                continue
            if spec.log:
                low = max(low, 1e-12)
                if high <= low:
                    continue
                # log-uniform in base-10
                a = self._rng.uniform(math.log10(low), math.log10(high))  # type: ignore[name-defined]
                params[spec.name] = float(10 ** a)
            else:
                params[spec.name] = float(self._rng.uniform(low, high))
        return params

    # ------------------------------------------------------------------ spaces
    def _default_space(self, metric: str) -> Sequence[ParamSpec]:
        metric = str(metric or "").strip().lower()
        if "perception" in metric:
            return (
                ParamSpec("predictive.reconstruction_lr", "float", low=1e-5, high=0.2, log=True),
                ParamSpec("predictive.prediction_lr", "float", low=1e-5, high=0.2, log=True),
            )
        # decision / generic: policy + exploration knobs
        return (
            ParamSpec("imitation.lr", "float", low=1e-4, high=0.2, log=True),
            ParamSpec("imitation.inference_uniform_mix", "float", low=0.0, high=0.25, log=False),
            ParamSpec("runtime.big_brain", "categorical", choices=(True, False)),
        )

    def _parse_space(self, raw: Any) -> Sequence[ParamSpec]:
        if isinstance(raw, (list, tuple)):
            specs: list[ParamSpec] = []
            for item in raw:
                if isinstance(item, ParamSpec):
                    specs.append(item)
                    continue
                if not isinstance(item, Mapping):
                    continue
                name = str(item.get("name") or "").strip()
                kind = str(item.get("kind") or item.get("type") or "float").strip().lower()
                if not name:
                    continue
                if kind == "categorical":
                    choices = item.get("choices")
                    if isinstance(choices, (list, tuple, set)) and choices:
                        specs.append(ParamSpec(name=name, kind="categorical", choices=tuple(choices)))
                    continue
                low = item.get("low")
                high = item.get("high")
                if low is None or high is None:
                    continue
                specs.append(
                    ParamSpec(
                        name=name,
                        kind="int" if kind == "int" else "float",
                        low=float(low),
                        high=float(high),
                        log=bool(item.get("log", False)),
                        step=float(item["step"]) if item.get("step") is not None else None,
                    )
                )
            return tuple(specs) if specs else self._default_space("default")
        return self._default_space("default")


__all__ = ["AutoMLManager", "ParamSpec"]
