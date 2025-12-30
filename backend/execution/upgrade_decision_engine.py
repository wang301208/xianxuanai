from __future__ import annotations

"""High-level upgrade decision engine (rules + optional learning).

The runtime already contains several self-improvement primitives:
- Module acquisition (discover/suggest/install): `ModuleAcquisitionManager`
- AutoML hyper-parameter search: `AutoMLManager`
- Architecture evolution: `HybridArchitectureManager` (via AdaptiveResourceController)

This component sits above them and answers:
  *When should we upgrade and in what direction?*

It consumes task outcomes + plan signals and emits conservative upgrade requests:
- `module.acquisition.request` (suggest new capability/module)
- `automl.request` (search hyper-params when performance stagnates)
- `upgrade.architecture.request` (ask runtime to run an architecture-evolution step)

By default it is disabled and uses rule-based triggers. An optional contextual
bandit (LinUCB) can be enabled to learn which upgrade action tends to improve
the observed success rate for the current context.
"""

import logging
import os
import random
import re
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Mapping, Optional, Sequence

try:  # optional in some deployments
    from events import EventBus  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    EventBus = None  # type: ignore

try:  # optional for learning policy
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy may be unavailable in minimal envs
    np = None  # type: ignore

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


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


_CLUSTER_KEYWORDS: Dict[str, Sequence[str]] = {
    "vision": ("image", "png", "jpeg", "jpg", "ocr", "pillow", "pil", "opencv", "cv2", "torchvision"),
    "pdf": ("pdf", "pypdf", "pdfplumber"),
    "audio": ("audio", "speech", "asr", "whisper"),
    "web": ("http", "https", "requests", "timeout", "connection", "429", "rate limit"),
    "database": ("sql", "sqlite", "postgres", "mysql", "vector", "embedding", "chromadb", "qdrant", "weaviate"),
}


def _flatten_failure_text(event: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("name", "category", "status", "error"):
        value = event.get(key)
        if value is None:
            continue
        parts.append(str(value))

    metadata = event.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("query", "goal", "reason", "task", "tasks"):
            value = metadata.get(key)
            if value is None:
                continue
            parts.append(str(value))

    autofix = event.get("autofix")
    if isinstance(autofix, Mapping):
        analysis = autofix.get("analysis")
        if isinstance(analysis, Mapping):
            parts.append(str(analysis.get("summary") or ""))
            parts.append(str(analysis.get("root_cause") or ""))
            parts.append(str(analysis.get("category") or ""))

    return " ".join(p for p in parts if p).lower()


def _infer_failure_cluster(text: str) -> str | None:
    blob = str(text or "").lower()
    if not blob:
        return None
    for cluster, keywords in _CLUSTER_KEYWORDS.items():
        for kw in keywords:
            if kw in blob:
                return cluster
    return None


@dataclass(frozen=True)
class UpgradeDecision:
    action: str
    reason: str
    score: float | None = None
    payload: Dict[str, Any] | None = None

    def to_event(self) -> Dict[str, Any]:
        return {
            "time": time.time(),
            "action": self.action,
            "reason": self.reason,
            "score": self.score,
            "payload": dict(self.payload or {}),
        }


class LinUCBBandit:
    """Minimal contextual bandit using LinUCB (per-arm ridge regression)."""

    def __init__(
        self,
        *,
        actions: Sequence[str],
        dim: int,
        alpha: float = 0.8,
        epsilon: float = 0.05,
        seed: int | None = None,
    ) -> None:
        if np is None:
            raise RuntimeError("numpy is required for LinUCBBandit")
        self.actions = [str(a) for a in actions if str(a)]
        if not self.actions:
            raise ValueError("actions must be non-empty")
        self.dim = int(max(1, dim))
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self._rng = random.Random(seed)

        self._A: Dict[str, "np.ndarray"] = {a: np.eye(self.dim, dtype=np.float64) for a in self.actions}
        self._b: Dict[str, "np.ndarray"] = {a: np.zeros(self.dim, dtype=np.float64) for a in self.actions}

    def choose(self, x: "np.ndarray") -> tuple[str, float]:
        if float(self._rng.random()) < max(0.0, float(self.epsilon)):
            action = self._rng.choice(self.actions)
            return str(action), 0.0

        best_action = self.actions[0]
        best_score = -float("inf")
        for action in self.actions:
            A = self._A[action]
            b = self._b[action]
            inv = np.linalg.inv(A)
            theta = inv @ b
            mean = float(theta @ x)
            bonus = float(self.alpha) * float(np.sqrt(max(0.0, x @ inv @ x)))
            score = mean + bonus
            if score > best_score:
                best_score = score
                best_action = action
        return best_action, float(best_score)

    def update(self, *, action: str, x: "np.ndarray", reward: float) -> None:
        token = str(action)
        if token not in self._A:
            return
        r = float(reward)
        self._A[token] = self._A[token] + np.outer(x, x)
        self._b[token] = self._b[token] + r * x


class UpgradeDecisionEngine:
    """Decide when to request upgrades and which direction to pursue."""

    ACTION_NONE = "none"
    ACTION_MODULE_ACQUISITION = "module_acquisition"
    ACTION_AUTOML = "automl"
    ACTION_ARCHITECTURE = "architecture_evolve"

    def __init__(
        self,
        *,
        event_bus: EventBus,
        enabled: bool | None = None,
        learning_enabled: bool | None = None,
        cooldown_secs: float | None = None,
        window_size: int | None = None,
        failure_threshold: int | None = None,
        same_cluster_threshold: int | None = None,
        stagnation_window: int | None = None,
        stagnation_min_delta: float | None = None,
        success_rate_target: float | None = None,
        reward_window_tasks: int | None = None,
        bandit_alpha: float | None = None,
        bandit_epsilon: float | None = None,
        logger_: Optional[logging.Logger] = None,
    ) -> None:
        if event_bus is None:
            raise ValueError("event_bus is required")
        self._bus = event_bus
        self._logger = logger_ or logger

        self._enabled = _env_bool("UPGRADE_DECISION_ENABLED", False) if enabled is None else bool(enabled)
        self._learning_enabled = (
            _env_bool("UPGRADE_DECISION_LEARNING", False) if learning_enabled is None else bool(learning_enabled)
        )
        self._cooldown_secs = (
            _env_float("UPGRADE_DECISION_COOLDOWN_SECS", 300.0) if cooldown_secs is None else float(cooldown_secs)
        )
        self._window_size = _env_int("UPGRADE_DECISION_WINDOW", 64) if window_size is None else int(window_size)
        self._failure_threshold = (
            _env_int("UPGRADE_DECISION_FAILURE_THRESHOLD", 4)
            if failure_threshold is None
            else int(failure_threshold)
        )
        self._same_cluster_threshold = (
            _env_int("UPGRADE_DECISION_SAME_CLUSTER_THRESHOLD", 3)
            if same_cluster_threshold is None
            else int(same_cluster_threshold)
        )
        self._stagnation_window = (
            _env_int("UPGRADE_DECISION_STAGNATION_WINDOW", 24)
            if stagnation_window is None
            else int(stagnation_window)
        )
        self._stagnation_min_delta = (
            _env_float("UPGRADE_DECISION_STAGNATION_MIN_DELTA", 0.02)
            if stagnation_min_delta is None
            else float(stagnation_min_delta)
        )
        self._success_rate_target = (
            _env_float("UPGRADE_DECISION_SUCCESS_RATE_TARGET", 0.7)
            if success_rate_target is None
            else float(success_rate_target)
        )
        self._reward_window_tasks = (
            _env_int("UPGRADE_DECISION_REWARD_WINDOW_TASKS", 16)
            if reward_window_tasks is None
            else int(reward_window_tasks)
        )

        self._last_decision_ts = 0.0
        self._outcomes: Deque[Dict[str, Any]] = deque(maxlen=max(10, int(self._window_size)))
        self._pending: Deque[Dict[str, Any]] = deque(maxlen=64)
        self._latest_plan: Dict[str, Any] = {}
        self._latest_knowledge_gaps: Sequence[str] = ()

        self._bandit: LinUCBBandit | None = None
        if self._enabled and self._learning_enabled and np is not None:
            alpha = _env_float("UPGRADE_DECISION_BANDIT_ALPHA", 0.8) if bandit_alpha is None else float(bandit_alpha)
            epsilon = (
                _env_float("UPGRADE_DECISION_BANDIT_EPSILON", 0.05)
                if bandit_epsilon is None
                else float(bandit_epsilon)
            )
            try:
                self._bandit = LinUCBBandit(
                    actions=[
                        self.ACTION_NONE,
                        self.ACTION_MODULE_ACQUISITION,
                        self.ACTION_AUTOML,
                        self.ACTION_ARCHITECTURE,
                    ],
                    dim=8,
                    alpha=alpha,
                    epsilon=epsilon,
                    seed=_env_int("UPGRADE_DECISION_SEED", 0),
                )
            except Exception:
                self._bandit = None

        self._subscriptions: list[Callable[[], None]] = [
            self._bus.subscribe("task_manager.task_completed", self._on_task_completed),
            self._bus.subscribe("planner.plan_ready", self._on_plan_ready),
            self._bus.subscribe("learning.cycle_completed", self._on_learning_cycle),
        ]

    def close(self) -> None:
        subs = list(self._subscriptions)
        self._subscriptions.clear()
        for cancel in subs:
            try:
                cancel()
            except Exception:
                continue

    # ------------------------------------------------------------------ public helpers
    def status(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self._enabled),
            "learning_enabled": bool(self._learning_enabled and self._bandit is not None),
            "window": int(self._window_size),
            "outcomes": int(len(self._outcomes)),
            "pending": int(len(self._pending)),
            "last_decision_ts": float(self._last_decision_ts),
        }

    # ------------------------------------------------------------------ decision logic
    def _cooldown_active(self, now: float) -> bool:
        if self._cooldown_secs <= 0:
            return False
        return (now - float(self._last_decision_ts)) < float(self._cooldown_secs)

    def _success_rate(self, *, tail: int) -> float | None:
        if not self._outcomes:
            return None
        tail = int(max(1, tail))
        recent = list(self._outcomes)[-tail:]
        if not recent:
            return None
        ok = sum(1 for r in recent if r.get("ok"))
        return ok / float(len(recent))

    def _dominant_failure_cluster(self, *, tail: int) -> tuple[str, int] | None:
        tail = int(max(1, tail))
        recent = list(self._outcomes)[-tail:]
        failures = [r for r in recent if not r.get("ok") and r.get("cluster")]
        if not failures:
            return None
        counts = Counter(str(r["cluster"]) for r in failures)
        cluster, count = counts.most_common(1)[0]
        return cluster, int(count)

    def _stagnating(self) -> bool:
        window = int(max(2, self._stagnation_window))
        if len(self._outcomes) < window * 2:
            return False
        older = self._success_rate(tail=window * 2)
        recent = self._success_rate(tail=window)
        if older is None or recent is None:
            return False
        # Approximate trend as (recent - older baseline of 2x window)
        delta = float(recent) - float(older)
        return abs(delta) < float(self._stagnation_min_delta)

    def _build_features(self) -> "np.ndarray | None":
        if np is None:
            return None
        recent_success = self._success_rate(tail=min(len(self._outcomes) or 1, self._stagnation_window)) or 0.0
        dominant = self._dominant_failure_cluster(tail=min(len(self._outcomes) or 1, self._stagnation_window))
        dominant_failures = float(dominant[1]) if dominant is not None else 0.0
        failures = sum(1 for r in self._outcomes if not r.get("ok"))
        total = len(self._outcomes)
        failure_rate = float(failures) / float(max(1, total))
        knowledge_gaps = float(len(self._latest_knowledge_gaps)) if self._latest_knowledge_gaps else 0.0
        plan_tasks = self._latest_plan.get("tasks") or []
        task_count = float(len(plan_tasks)) if isinstance(plan_tasks, list) else 0.0
        pending = float(len(self._pending))
        # Features: bias + simple scalars (keep dim fixed).
        return np.asarray(
            [
                1.0,
                float(recent_success),
                float(failure_rate),
                float(dominant_failures) / float(max(1.0, self._same_cluster_threshold)),
                float(knowledge_gaps) / 10.0,
                float(task_count) / 10.0,
                float(pending) / 10.0,
                1.0 if self._stagnating() else 0.0,
            ],
            dtype=np.float64,
        )

    def _rule_decision(self) -> UpgradeDecision | None:
        # Rule 1: repeated failures of a single cluster -> request new capability.
        dominant = self._dominant_failure_cluster(tail=self._window_size)
        if dominant is not None:
            cluster, count = dominant
            if count >= int(self._same_cluster_threshold):
                query = str(cluster)
                return UpgradeDecision(
                    action=self.ACTION_MODULE_ACQUISITION,
                    reason=f"repeated_failure_cluster:{cluster}:{count}",
                    payload={"query": query, "cluster": cluster, "count": int(count)},
                )

        # Rule 2: overall failure pressure -> request a general capability search.
        failures = sum(1 for r in self._outcomes if not r.get("ok"))
        if failures >= int(self._failure_threshold):
            return UpgradeDecision(
                action=self.ACTION_MODULE_ACQUISITION,
                reason=f"failure_pressure:{failures}",
                payload={"query": "python library to improve task success", "failures": int(failures)},
            )

        # Rule 3: performance stagnation below target -> request architecture/AutoML probe.
        success = self._success_rate(tail=max(2, self._stagnation_window))
        if success is not None and float(success) < float(self._success_rate_target) and self._stagnating():
            # Prefer architecture evolution; AutoML is also viable.
            return UpgradeDecision(
                action=self.ACTION_ARCHITECTURE,
                reason=f"success_rate_stagnation:{success:.3f}",
                payload={"success_rate": float(success), "target": float(self._success_rate_target)},
            )

        return None

    def _select_decision(self) -> UpgradeDecision | None:
        rule = self._rule_decision()
        if rule is not None:
            return rule
        if self._bandit is None or np is None:
            return None
        x = self._build_features()
        if x is None:
            return None
        action, score = self._bandit.choose(x)
        if action == self.ACTION_NONE:
            return None
        return UpgradeDecision(action=action, reason="bandit", score=float(score), payload={"features": x.tolist()})

    def _emit_decision(self, decision: UpgradeDecision) -> None:
        try:
            self._bus.publish("upgrade.decision", decision.to_event())
        except Exception:
            pass

    def _execute_decision(self, decision: UpgradeDecision) -> None:
        payload = dict(decision.payload or {})
        now = time.time()

        if decision.action == self.ACTION_MODULE_ACQUISITION:
            query = payload.get("query") or ""
            if not isinstance(query, str) or not query.strip():
                query = "python module for missing capability"
            try:
                self._bus.publish(
                    "module.acquisition.request",
                    {
                        "time": now,
                        "query": str(query),
                        "reason": decision.reason,
                        "context": {
                            "latest_plan": dict(self._latest_plan or {}),
                            "knowledge_gaps": list(self._latest_knowledge_gaps or []),
                        },
                    },
                )
            except Exception:
                pass

        elif decision.action == self.ACTION_AUTOML:
            try:
                self._bus.publish(
                    "automl.request",
                    {
                        "time": now,
                        "metric": "decision_success_rate",
                        "direction": "increase",
                        "source": "upgrade_decision",
                        "reason": decision.reason,
                    },
                )
            except Exception:
                pass

        elif decision.action == self.ACTION_ARCHITECTURE:
            try:
                self._bus.publish(
                    "upgrade.architecture.request",
                    {
                        "time": now,
                        "source": "upgrade_decision",
                        "reason": decision.reason,
                        "steps": 1,
                        "hint": payload,
                    },
                )
            except Exception:
                pass

        # Track reward baseline for learning.
        if self._bandit is not None and np is not None:
            x = self._build_features()
            if x is not None:
                baseline_total = len(self._outcomes)
                baseline_success = sum(1 for r in self._outcomes if r.get("ok"))
                self._pending.append(
                    {
                        "time": now,
                        "action": decision.action,
                        "features": x,
                        "baseline_total": int(baseline_total),
                        "baseline_success": int(baseline_success),
                        "settle_after": int(baseline_total) + int(max(1, self._reward_window_tasks)),
                    }
                )

    def _update_bandit_rewards(self) -> None:
        if self._bandit is None or np is None:
            return
        if not self._pending:
            return
        total = len(self._outcomes)
        if total <= 0:
            return
        success_total = sum(1 for r in self._outcomes if r.get("ok"))

        remaining: Deque[Dict[str, Any]] = deque(maxlen=self._pending.maxlen)
        for record in list(self._pending):
            settle_after = int(record.get("settle_after", total + 1))
            if total < settle_after:
                remaining.append(record)
                continue
            baseline_total = int(record.get("baseline_total", 0))
            baseline_success = int(record.get("baseline_success", 0))
            delta_total = max(1, total - baseline_total)
            delta_success = success_total - baseline_success
            # Reward: delta success-rate in the post-window.
            reward = float(delta_success) / float(delta_total)
            action = str(record.get("action") or "")
            x = record.get("features")
            if isinstance(x, np.ndarray):
                try:
                    self._bandit.update(action=action, x=x, reward=reward)
                except Exception:
                    pass
        self._pending = remaining

    # ------------------------------------------------------------------ event handlers
    async def _on_task_completed(self, event: Dict[str, Any]) -> None:
        if not self._enabled or not isinstance(event, Mapping):
            return
        status = str(event.get("status") or "").strip().lower()
        ok = status in {"completed", "success", "ok"}
        blob = _flatten_failure_text(event)
        cluster = _infer_failure_cluster(blob) if not ok else None
        self._outcomes.append(
            {
                "time": float(event.get("time", time.time()) or time.time()),
                "ok": bool(ok),
                "cluster": cluster,
            }
        )
        self._update_bandit_rewards()
        if ok:
            return
        now = time.time()
        if self._cooldown_active(now):
            return
        decision = self._select_decision()
        if decision is None:
            return
        self._last_decision_ts = now
        self._emit_decision(decision)
        self._execute_decision(decision)

    async def _on_plan_ready(self, event: Dict[str, Any]) -> None:
        if not self._enabled or not isinstance(event, Mapping):
            return
        self._latest_plan = dict(event)
        gaps = event.get("knowledge_gap_domains") or event.get("knowledge_gaps") or ()
        if isinstance(gaps, (list, tuple, set)):
            self._latest_knowledge_gaps = [str(g) for g in gaps if str(g)]
        now = time.time()
        if self._cooldown_active(now):
            return
        decision = self._select_decision()
        if decision is None:
            return
        self._last_decision_ts = now
        self._emit_decision(decision)
        self._execute_decision(decision)

    async def _on_learning_cycle(self, event: Dict[str, Any]) -> None:
        if not self._enabled or not isinstance(event, Mapping):
            return
        # Lightweight periodic evaluation point (e.g. stagnation detection).
        now = time.time()
        if self._cooldown_active(now):
            return
        decision = self._select_decision()
        if decision is None:
            return
        self._last_decision_ts = now
        self._emit_decision(decision)
        self._execute_decision(decision)


__all__ = ["UpgradeDecisionEngine", "UpgradeDecision", "LinUCBBandit"]
