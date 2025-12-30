"""Adaptive responses to sustained resource pressure."""
from __future__ import annotations

import logging
import math
import os
import random
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Iterable, Mapping, Optional, Sequence, Tuple

try:  # Optional dependency during minimal test environments
    from events import EventBus  # type: ignore
except Exception:  # pragma: no cover - optional import
    EventBus = None  # type: ignore

try:  # Optional evolutionary imports
    from modules.evolution.evolving_cognitive_architecture import (
        GAConfig,
        GeneticAlgorithm,
    )
except Exception:  # pragma: no cover - fallback used in tests
    GAConfig = None  # type: ignore

    class GeneticAlgorithm:  # type: ignore
        """Lightweight fallback GA supporting evolve(seed)."""

        def __init__(self, fitness_fn: Callable[[Dict[str, float]], float], config: Any | None = None, seed: int | None = None) -> None:
            self.fitness_fn = fitness_fn
            self._rng = random.Random(seed)
            cfg = config or type("Cfg", (), {"population_size": 8, "generations": 3, "mutation_rate": 0.3, "mutation_sigma": 0.1})()
            self.config = cfg

        def _mutate(self, candidate: Dict[str, float]) -> Dict[str, float]:
            mutated = candidate.copy()
            for key in mutated:
                if self._rng.random() < self.config.mutation_rate:
                    mutated[key] += self._rng.gauss(0.0, getattr(self.config, "mutation_sigma", 0.1))
            return mutated

        def evolve(self, seed: Dict[str, float]) -> Tuple[Dict[str, float], float, list[Tuple[Dict[str, float], float]]]:
            best = seed.copy()
            best_score = self.fitness_fn(best)
            history: list[Tuple[Dict[str, float], float]] = [(best.copy(), best_score)]
            population = [self._mutate(seed) for _ in range(getattr(self.config, "population_size", 8))]
            for _ in range(getattr(self.config, "generations", 3)):
                scored = [(cand, self.fitness_fn(cand)) for cand in population]
                scored.sort(key=lambda item: item[1], reverse=True)
                if scored[0][1] > best_score:
                    best, best_score = scored[0][0].copy(), scored[0][1]
                history.extend((cand.copy(), score) for cand, score in scored[:2])
                population = [self._mutate(best) for _ in population]
            return best, best_score, history

    class GAConfig:  # type: ignore
        def __init__(self, population_size: int = 8, generations: int = 3, mutation_rate: float = 0.3, mutation_sigma: float = 0.15) -> None:
            self.population_size = population_size
            self.generations = generations
            self.mutation_rate = mutation_rate
            self.mutation_sigma = mutation_sigma

try:  # Optional PSO optimisation
    from backend.pso import pso
except Exception:  # pragma: no cover - optional
    pso = None  # type: ignore[assignment]

try:
    from backend.monitoring import MultiMetricMonitor
except Exception:  # pragma: no cover - monitor optional
    MultiMetricMonitor = None  # type: ignore[assignment]

try:
    from backend.execution.self_improvement import SelfImprovementManager
except Exception:  # pragma: no cover - optional
    SelfImprovementManager = None  # type: ignore[assignment]


class InternalFeedbackEvaluator:
    """Detect weak spots from streaming metrics and produce structured feedback."""

    def __init__(
        self,
        *,
        perception_conf_threshold: float = 0.5,
        perception_error_threshold: float = 0.35,
        decision_success_threshold: float = 0.55,
        decision_reward_threshold: float = 0.1,
        memory_hit_rate_threshold: float = 0.05,
        latency_warn_threshold_ms: float = 500.0,
    ) -> None:
        self.perception_conf_threshold = perception_conf_threshold
        self.perception_error_threshold = perception_error_threshold
        self.decision_success_threshold = decision_success_threshold
        self.decision_reward_threshold = decision_reward_threshold
        self.memory_hit_rate_threshold = memory_hit_rate_threshold
        self.latency_warn_threshold_ms = latency_warn_threshold_ms

    def evaluate(self, metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
        feedback: list[dict[str, Any]] = []
        conf = metrics.get("perception_confidence_avg")
        if conf is not None:
            try:
                if float(conf) < self.perception_conf_threshold:
                    feedback.append(
                        {
                            "module": "perception",
                            "type": "low_confidence",
                            "severity": "warn",
                            "value": float(conf),
                            "threshold": self.perception_conf_threshold,
                        }
                    )
            except Exception:
                pass
        err = metrics.get("perception_prediction_error")
        if err is not None:
            try:
                if float(err) > self.perception_error_threshold:
                    feedback.append(
                        {
                            "module": "perception",
                            "type": "high_prediction_error",
                            "severity": "warn",
                            "value": float(err),
                            "threshold": self.perception_error_threshold,
                        }
                    )
            except Exception:
                pass
        success_rate = metrics.get("decision_success_rate")
        if success_rate is not None:
            try:
                if float(success_rate) < self.decision_success_threshold:
                    feedback.append(
                        {
                            "module": "decision",
                            "type": "low_reward",
                            "severity": "warn",
                            "value": float(success_rate),
                            "threshold": self.decision_success_threshold,
                        }
                    )
            except Exception:
                pass
        avg_reward = metrics.get("decision_reward_avg")
        if avg_reward is not None:
            try:
                if float(avg_reward) < self.decision_reward_threshold:
                    feedback.append(
                        {
                            "module": "decision",
                            "type": "low_reward_avg",
                            "severity": "warn",
                            "value": float(avg_reward),
                            "threshold": self.decision_reward_threshold,
                        }
                    )
            except Exception:
                pass
        mem_rate = metrics.get("memory_hit_rate")
        if mem_rate is not None:
            try:
                if float(mem_rate) < self.memory_hit_rate_threshold:
                    feedback.append(
                        {
                            "module": "memory",
                            "type": "low_hit_rate",
                            "severity": "warn",
                            "value": float(mem_rate),
                            "threshold": self.memory_hit_rate_threshold,
                        }
                    )
            except Exception:
                pass
        module_latency = metrics.get("module_latency_max")
        if module_latency is not None:
            try:
                if float(module_latency) > self.latency_warn_threshold_ms:
                    feedback.append(
                        {
                            "module": "module_latency",
                            "type": "high_latency",
                            "severity": "warn",
                            "value": float(module_latency),
                            "threshold": self.latency_warn_threshold_ms,
                        }
                    )
            except Exception:
                pass
        return feedback


class MetaLearningTuner:
    """Lightweight meta-learner that nudges exploration/learning rates based on recent returns."""

    def __init__(
        self,
        target_success: float = 0.7,
        high_success: float = 0.9,
        exploration_step: float = 0.05,
        learning_rate_step: float = 0.01,
        window: int = 6,
    ) -> None:
        self.target_success = target_success
        self.high_success = high_success
        self.exploration_step = exploration_step
        self.learning_rate_step = learning_rate_step
        self.success_window: Deque[float] = deque(maxlen=max(2, window))
        self.reward_window: Deque[float] = deque(maxlen=max(2, window))

    def observe(self, metrics: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """Return meta-adjustment suggestions when trends warrant adaptation."""

        success = metrics.get("decision_success_rate")
        reward = metrics.get("decision_reward_avg", metrics.get("reward"))
        if success is not None:
            try:
                self.success_window.append(float(success))
            except Exception:
                pass
        if reward is not None:
            try:
                self.reward_window.append(float(reward))
            except Exception:
                pass

        if len(self.success_window) < max(2, self.success_window.maxlen // 2):
            return None

        avg_success = sum(self.success_window) / max(len(self.success_window), 1)
        avg_reward = (
            sum(self.reward_window) / max(len(self.reward_window), 1)
            if self.reward_window
            else 0.0
        )

        adjustments: Dict[str, float] = {}
        exploration_hint: Optional[str] = None
        if avg_success < self.target_success:
            adjustments["exploration_rate"] = self.exploration_step
            adjustments["learning_rate"] = self.learning_rate_step
            exploration_hint = "increase"
        elif avg_success > self.high_success and avg_reward >= 0.0:
            adjustments["exploration_rate"] = -self.exploration_step
            exploration_hint = "decrease"
        else:
            return None

        return {
            "reason": "meta_learning",
            "metrics": {"avg_success": avg_success, "avg_reward": avg_reward},
            "adjustments": adjustments,
            "suggested": {"exploration": exploration_hint},
        }


@dataclass
class MetricTunerConfig:
    """Configuration for the metric-driven tuner."""

    strategy: str = "evolutionary"
    cooldown: float = 180.0
    reward_drop_tolerance: float = 0.05
    rollback_patience: int = 3
    exploration_sigma: float = 0.05
    learning_rate_sigma: float = 0.05
    module_toggle_prob: float = 0.25


class MetricMapTuner:
    """Explore small perturbations guided by the observed metric map."""

    def __init__(
        self,
        config: MetricTunerConfig,
        *,
        reward_fn: Callable[[Mapping[str, Any]], float | None],
        rng: Optional[random.Random] = None,
    ) -> None:
        self._config = config
        self._reward_fn = reward_fn
        self._rng = rng or random.Random()
        self._last_suggestion_ts = 0.0
        self._history: Deque[tuple[float, Dict[str, Any]]] = deque(maxlen=12)
        self._best: tuple[float, Dict[str, Any]] | None = None
        self._active_assignment: Dict[str, Any] | None = None

    def _probe(self, config: Any, name: str, default: float = 0.0) -> float:
        try:
            return float(getattr(config, name))
        except Exception:
            pass
        if isinstance(config, dict) and name in config:
            try:
                return float(config[name])
            except Exception:
                return default
        return default

    def _apply_assignment(self, config: Any, assignments: Mapping[str, Any]) -> None:
        for key, value in assignments.items():
            try:
                setattr(config, key, value)
            except Exception:
                if isinstance(config, dict):
                    config[key] = value

    def _suggest_toggle(self, config: Any, key: str) -> Optional[bool]:
        if not hasattr(config, key) and not (isinstance(config, dict) and key in config):
            return None
        return bool(self._rng.random() < 0.5)

    def _generate_assignment(self, metrics: Mapping[str, Any], config: Any) -> Dict[str, Any]:
        assignments: Dict[str, Any] = {}
        strategy = str(self._config.strategy or "").strip().lower()
        if strategy not in {"evolutionary", "optuna-lite", "bayesian", "bayesian_optimization"}:
            return assignments

        if strategy in {"optuna-lite", "bayesian", "bayesian_optimization"}:
            assignments.update(self._generate_bayesian_assignment(metrics, config))
            if not assignments:
                assignments.update(self._generate_perturbation_assignment(metrics, config))
        else:
            assignments.update(self._generate_perturbation_assignment(metrics, config))

        for toggle_key in ("big_brain", "prefer_structured_planner", "enable_curiosity_feedback"):
            if self._rng.random() < self._config.module_toggle_prob:
                toggle_value = self._suggest_toggle(config, toggle_key)
                if toggle_value is not None:
                    assignments[toggle_key] = toggle_value

        return assignments

    def _generate_perturbation_assignment(self, metrics: Mapping[str, Any], config: Any) -> Dict[str, Any]:
        base_explore = self._probe(config, "policy_exploration_rate", metrics.get("policy_exploration_rate", 0.1))
        base_lr = self._probe(config, "policy_learning_rate", metrics.get("policy_learning_rate", 0.05))
        explore = max(0.0, min(1.0, base_explore + self._rng.gauss(0.0, self._config.exploration_sigma)))
        lr = max(1e-5, min(1.0, base_lr * (1.0 + self._rng.gauss(0.0, self._config.learning_rate_sigma))))
        return {"policy_exploration_rate": explore, "policy_learning_rate": lr}

    def _generate_bayesian_assignment(self, metrics: Mapping[str, Any], config: Any) -> Dict[str, Any]:
        """Suggest hyper-parameters via a lightweight Bayesian optimisation loop.

        Notes:
        - Focuses on numeric hyper-parameters (exploration/lr) only.
        - Uses a tiny GP surrogate + expected improvement, optimised by random search.
        - Falls back to perturbation when insufficient history exists.
        """

        try:
            import numpy as np  # type: ignore
        except Exception:
            return {}

        xs: list[list[float]] = []
        ys: list[float] = []
        for reward, assignment in list(self._history):
            if not isinstance(assignment, Mapping):
                continue
            exp = assignment.get("policy_exploration_rate")
            lr = assignment.get("policy_learning_rate")
            try:
                exp_v = float(exp)
                lr_v = float(lr)
            except Exception:
                continue
            if lr_v <= 0:
                continue
            exp_v = max(0.0, min(1.0, exp_v))
            xs.append([exp_v, self._lr_to_feature(lr_v)])
            ys.append(float(reward))

        if len(xs) < 4:
            return {}

        x_train = np.asarray(xs, dtype=np.float64)
        y_train = np.asarray(ys, dtype=np.float64)
        y_mean = float(np.mean(y_train))
        y_std = float(np.std(y_train))
        if y_std > 1e-8:
            y_norm = (y_train - y_mean) / y_std
        else:
            y_norm = y_train - y_mean

        lengthscale = float(os.getenv("ADAPTIVE_TUNER_BO_LENGTHSCALE", "0.25") or 0.25)
        lengthscale = max(1e-3, lengthscale)
        noise = float(os.getenv("ADAPTIVE_TUNER_BO_NOISE", "1e-6") or 1e-6)
        noise = max(1e-12, noise)

        diff = x_train[:, None, :] - x_train[None, :, :]
        sqdist = np.sum(diff * diff, axis=-1)
        k = np.exp(-0.5 * sqdist / (lengthscale * lengthscale))
        k = k + noise * np.eye(len(x_train), dtype=np.float64)

        try:
            l = np.linalg.cholesky(k)
            alpha = np.linalg.solve(l.T, np.linalg.solve(l, y_norm))
        except Exception:
            return {}

        best_y = float(np.max(y_norm))
        xi = float(os.getenv("ADAPTIVE_TUNER_BO_XI", "0.01") or 0.01)
        xi = max(0.0, xi)

        candidates = int(os.getenv("ADAPTIVE_TUNER_BO_CANDIDATES", "64") or 64)
        candidates = max(16, int(candidates))
        explore_sigma = float(os.getenv("ADAPTIVE_TUNER_BO_SIGMA", "0.12") or 0.12)
        explore_sigma = max(1e-6, explore_sigma)

        best_idx = int(np.argmax(y_norm))
        best_x = x_train[best_idx]
        cand_points: list[list[float]] = []
        for _ in range(candidates):
            cand_points.append([self._rng.random(), self._rng.random()])
        for _ in range(max(8, candidates // 4)):
            cand_points.append(
                [
                    max(0.0, min(1.0, float(best_x[0]) + self._rng.gauss(0.0, explore_sigma))),
                    max(0.0, min(1.0, float(best_x[1]) + self._rng.gauss(0.0, explore_sigma))),
                ]
            )
        x_cand = np.asarray(cand_points, dtype=np.float64)

        mu, var = self._gp_predict(x_train, l, alpha, x_cand, lengthscale=lengthscale)
        std = np.sqrt(np.maximum(var, 1e-12))
        improvement = mu - best_y - xi
        z = improvement / std
        phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
        ei = improvement * cdf + std * phi
        try:
            best_cand = int(np.argmax(ei))
        except Exception:
            best_cand = 0

        exp_next = float(x_cand[best_cand, 0])
        lr_next = self._feature_to_lr(float(x_cand[best_cand, 1]))
        return {"policy_exploration_rate": exp_next, "policy_learning_rate": lr_next}

    @staticmethod
    def _gp_predict(x_train: Any, l: Any, alpha: Any, x_query: Any, *, lengthscale: float) -> tuple[Any, Any]:
        import numpy as np  # type: ignore

        diff = x_train[:, None, :] - x_query[None, :, :]
        sqdist = np.sum(diff * diff, axis=-1)
        k_star = np.exp(-0.5 * sqdist / (lengthscale * lengthscale))
        mu = k_star.T @ alpha
        v = np.linalg.solve(l, k_star)
        var = 1.0 - np.sum(v * v, axis=0)
        return mu, var

    @staticmethod
    def _lr_to_feature(lr: float) -> float:
        lr = max(1e-5, min(1.0, float(lr)))
        log_lr = math.log10(lr)
        return max(0.0, min(1.0, (log_lr + 5.0) / 5.0))

    @staticmethod
    def _feature_to_lr(feature: float) -> float:
        value = max(0.0, min(1.0, float(feature)))
        log_lr = value * 5.0 - 5.0
        return float(10 ** log_lr)

    def suggest(self, metrics: Mapping[str, Any], config: Any) -> Optional[Dict[str, Any]]:
        """Propose assignments using the latest metric map."""

        reward = self._reward_fn(metrics)
        if reward is not None:
            if self._active_assignment:
                snapshot = dict(self._active_assignment)
            else:
                snapshot = {
                    "policy_exploration_rate": self._probe(
                        config,
                        "policy_exploration_rate",
                        metrics.get("policy_exploration_rate", 0.1),
                    ),
                    "policy_learning_rate": self._probe(
                        config,
                        "policy_learning_rate",
                        metrics.get("policy_learning_rate", 0.05),
                    ),
                }
            self._history.append((float(reward), snapshot))
            if self._best is None or float(reward) > self._best[0]:
                self._best = (float(reward), dict(snapshot))

        now = time.time()
        if (
            reward is not None
            and self._best is not None
            and self._active_assignment
            and float(reward)
            < self._best[0] - abs(self._best[0]) * self._config.reward_drop_tolerance
            and len(self._history) >= self._config.rollback_patience
        ):
            rollback = dict(self._best[1])
            if rollback:
                self._apply_assignment(config, rollback)
                self._active_assignment = rollback
                self._last_suggestion_ts = now
                return {
                    "strategy": "rollback",
                    "assignments": rollback,
                    "reward": float(reward),
                    "best_reward": self._best[0],
                }

        if now - self._last_suggestion_ts < self._config.cooldown:
            return None

        assignments = self._generate_assignment(metrics, config)
        if not assignments:
            return None

        self._apply_assignment(config, assignments)
        self._active_assignment = dict(assignments)
        self._last_suggestion_ts = now
        return {
            "strategy": self._config.strategy,
            "assignments": assignments,
            "reward": float(reward) if reward is not None else None,
            "best_reward": self._best[0] if self._best is not None else None,
        }


class AdaptiveResourceController:
    """Apply mitigation strategies when system resources are constrained."""

    def __init__(
        self,
        *,
        config: Any,
        event_bus: Optional[EventBus],
        memory_router: Any | None,
        long_term_memory: Any | None,
        logger: Optional[logging.Logger] = None,
        memory_threshold: float = 85.0,
        memory_cooldown: float = 180.0,
        cpu_high_threshold: float = 85.0,
        cpu_recover_threshold: float = 60.0,
        mode_cooldown: float = 300.0,
        meta_policy: Optional["EpsilonGreedyMetaPolicy"] = None,
        architecture_manager: Optional["HybridArchitectureManager"] = None,
        reward_fn: Optional[Callable[[Mapping[str, Any]], float | None]] = None,
        rng_seed: Optional[int] = None,
        monitor: Optional["MultiMetricMonitor"] = None,
        **kwargs: Any,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._memory_router = memory_router
        self._long_term_memory = long_term_memory
        self._logger = logger or logging.getLogger(__name__)

        self._memory_threshold = memory_threshold
        self._memory_cooldown = max(0.0, memory_cooldown)
        self._cpu_high_threshold = cpu_high_threshold
        self._cpu_recover_threshold = cpu_recover_threshold
        self._mode_cooldown = max(0.0, mode_cooldown)

        self._last_memory_compaction = 0.0
        self._last_mode_change = 0.0
        self._initial_big_brain = bool(getattr(config, "big_brain", True))
        self._mode = "high" if self._initial_big_brain else "low"
        self._meta_policy = meta_policy
        self._architecture_manager = architecture_manager
        self._arch_auto_enabled = str(os.getenv("ARCH_EVOLUTION_AUTOMATIC", "1")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._arch_force_steps = 0
        self._arch_force_reason = ""
        self._arch_force_lock = threading.Lock()
        self._reward_fn = reward_fn or self._default_reward
        self._rng = random.Random(rng_seed)
        self._last_metrics: Optional[Mapping[str, Any]] = None
        self._module_metrics: Dict[str, Dict[str, list[float]]] = {}
        self._monitor = monitor
        self._extra_metrics: Dict[str, float] = {}
        self._feedback_evaluator = InternalFeedbackEvaluator()
        self._feedback_log: list[dict[str, Any]] = []
        self._meta_adjustment_provider = kwargs.pop("meta_adjustment_provider", None)
        self._retrain_callback = kwargs.pop("retrain_callback", None)
        self._module_swapper = kwargs.pop("module_swapper", None)
        self._resource_optimizer = kwargs.pop("resource_optimizer", None)
        self._self_improvement = (
            kwargs.pop("self_improvement_manager", None)
            if "self_improvement_manager" in kwargs
            else (SelfImprovementManager() if SelfImprovementManager is not None else None)
        )
        self._meta_tuner = MetaLearningTuner()
        tuner_cfg = self._parse_tuner_config(getattr(config, "adaptive_tuner", None))
        self._metric_tuner = (
            MetricMapTuner(tuner_cfg, reward_fn=self._reward_fn, rng=self._rng)
            if tuner_cfg is not None
            else None
        )

    # ------------------------------------------------------------------
    def update(
        self,
        avg_cpu: float,
        avg_memory: float,
        backlog: int,
        metrics: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Process the latest resource snapshot."""

        now = time.time()
        if avg_memory >= self._memory_threshold:
            self._handle_memory_pressure(now, avg_memory)

        if avg_cpu >= self._cpu_high_threshold and backlog > 0:
            self._enter_low_fidelity(now, avg_cpu, backlog)
        elif avg_cpu <= self._cpu_recover_threshold and backlog == 0:
            self._restore_high_fidelity(now, avg_cpu)

        metric_map: Dict[str, Any] = dict(metrics or {})
        metric_map.setdefault("avg_cpu", avg_cpu)
        metric_map.setdefault("avg_memory", avg_memory)
        metric_map.setdefault("backlog", backlog)
        module_stats = self._summarise_module_metrics()
        metric_map.update(module_stats)
        self._record_resource_metrics(avg_cpu, avg_memory)
        metric_map.update(self._extra_metrics)

        # Update bandit with reward from previous cycle before choosing new actions.
        if self._meta_policy and self._last_metrics is not None:
            reward = self._reward_fn(self._last_metrics)
            if reward is not None:
                self._meta_policy.record_outcome(float(reward))

        if self._meta_policy:
            assignments = self._meta_policy.apply(self._config, metric_map, rng=self._rng)
            if assignments:
                self._logger.info(
                    "Meta-parameters tuned: %s",
                    ", ".join(f"{key}={value}" for key, value in assignments.items()),
                )

        tuner_result = None
        if self._metric_tuner is not None:
            tuner_result = self._metric_tuner.suggest(metric_map, self._config)
            if tuner_result and tuner_result.get("assignments"):
                metric_map.setdefault("tuner_assignments", tuner_result["assignments"])
                self._logger.info(
                    "Adaptive tuner (%s) suggested: %s",
                    tuner_result.get("strategy"),
                    ", ".join(
                        f"{key}={value}" for key, value in tuner_result.get("assignments", {}).items()
                    ),
                )

        if self._architecture_manager is not None:
            force = False
            force_reason = ""
            with self._arch_force_lock:
                if int(self._arch_force_steps) > 0:
                    self._arch_force_steps = int(self._arch_force_steps) - 1
                    force = True
                    force_reason = str(self._arch_force_reason or "")
            if self._arch_auto_enabled or force:
                if force and force_reason:
                    metric_map.setdefault("arch_evolution_request_reason", force_reason)
                arch_event = self._architecture_manager.observe(metric_map)
            else:
                recorder = getattr(self._architecture_manager, "record", None)
                if callable(recorder):
                    try:
                        recorder(metric_map)
                    except Exception:
                        pass
                arch_event = None
            if arch_event is not None and self._event_bus is not None:
                self._event_bus.publish(
                    "resource.adaptation.architecture",
                    arch_event,
                )
                self._logger.info(
                    "Architecture evolved -> score %.3f",
                    arch_event["score"],
                )

        self._last_metrics = metric_map
        self._monitor_snapshot(metric_map)
        self._evaluate_feedback(metric_map)
        self._apply_meta_learning(metric_map)
        self._apply_feedback()
        self._adapt_goals(metric_map)

    def request_architecture_evolution(self, *, reason: str = "upgrade_decision", steps: int = 1) -> bool:
        """Request one or more architecture evolution steps.

        When `ARCH_EVOLUTION_AUTOMATIC=0`, evolution runs only after this is called.
        Returns `False` when no architecture manager is attached.
        """

        if self._architecture_manager is None:
            return False
        try:
            count = int(steps)
        except Exception:
            count = 1
        count = max(1, count)
        with self._arch_force_lock:
            self._arch_force_steps = max(int(self._arch_force_steps), count)
            self._arch_force_reason = str(reason or "upgrade_decision").strip() or "upgrade_decision"
        return True

    # ------------------------------------------------------------------ #
    def _parse_tuner_config(self, raw: Any) -> Optional[MetricTunerConfig]:
        if raw is None or raw is False:
            return None

        cfg = MetricTunerConfig()
        if isinstance(raw, Mapping):
            strategy = raw.get("strategy", cfg.strategy)
            cfg.strategy = "none" if strategy is None else str(strategy)
            cfg.cooldown = float(raw.get("cooldown", cfg.cooldown))
            cfg.reward_drop_tolerance = float(raw.get("reward_drop_tolerance", cfg.reward_drop_tolerance))
            cfg.rollback_patience = int(raw.get("rollback_patience", cfg.rollback_patience))
            cfg.exploration_sigma = float(raw.get("exploration_sigma", cfg.exploration_sigma))
            cfg.learning_rate_sigma = float(raw.get("learning_rate_sigma", cfg.learning_rate_sigma))
            cfg.module_toggle_prob = float(raw.get("module_toggle_prob", cfg.module_toggle_prob))
        elif isinstance(raw, str):
            cfg.strategy = raw
        elif raw is True:
            cfg.strategy = cfg.strategy or "evolutionary"

        if cfg.strategy in {"none", "off"}:
            return None
        return cfg

    # ------------------------------------------------------------------ #
    def _record_resource_metrics(self, cpu: float, memory: float) -> None:
        if self._monitor is None:
            return
        try:
            self._monitor.log_resource()
            self._monitor.log_snapshot({"avg_cpu": float(cpu), "avg_memory": float(memory)})
        except Exception:  # pragma: no cover - monitoring is best-effort
            pass

    # ------------------------------------------------------------------ #
    def record_module_metric(
        self,
        name: str,
        *,
        latency: float | None = None,
        throughput: float | None = None,
    ) -> None:
        """Record per-module performance signals for fitness shaping."""

        stats = self._module_metrics.setdefault(name, {"latency": [], "throughput": []})
        if latency is not None:
            try:
                stats["latency"].append(float(latency))
            except Exception:  # pragma: no cover - defensive
                pass
        if throughput is not None:
            try:
                stats["throughput"].append(float(throughput))
            except Exception:  # pragma: no cover
                pass

    def _summarise_module_metrics(self) -> Dict[str, Any]:
        """Return aggregate module stats and clear buffers."""

        if not self._module_metrics:
            return {}
        summary: Dict[str, Any] = {}
        max_latency = 0.0
        for name, stats in list(self._module_metrics.items()):
            latencies = stats.get("latency") or []
            throughputs = stats.get("throughput") or []
            if latencies:
                avg_lat = sum(latencies) / max(len(latencies), 1)
                summary[f"module_latency_{name}"] = avg_lat
                max_latency = max(max_latency, avg_lat)
            if throughputs:
                avg_thr = sum(throughputs) / max(len(throughputs), 1)
                summary[f"module_throughput_{name}"] = avg_thr
        if max_latency:
            summary["module_latency_max"] = max_latency
        # Reset after summarising to avoid unbounded growth
        self._module_metrics = {}
        return summary

    def _monitor_snapshot(self, metrics: Mapping[str, Any]) -> None:
        """Send metrics to the optional MultiMetricMonitor."""

        if self._monitor is None:
            return
        try:
            if "reward" in metrics:
                self._monitor.log_inference(float(metrics.get("reward", 0.0)))
            if "success_rate" in metrics:
                self._monitor.log_training(float(metrics.get("success_rate", 0.0)))
            try:
                self._monitor.log_resource()
            except Exception:
                pass
            scalar_metrics: Dict[str, float] = {}
            for key in (
                "avg_cpu",
                "avg_memory",
                "backlog",
                "module_latency_max",
                "module_throughput_goal_listener",
                "module_latency_goal_listener",
                "decision_reward_avg",
                "decision_success_rate",
            ):
                if key in metrics:
                    try:
                        scalar_metrics[key] = float(metrics[key])
                    except Exception:
                        continue
            if scalar_metrics:
                self._monitor.log_snapshot(scalar_metrics)
        except Exception:  # pragma: no cover - monitoring is best-effort
            pass

    def record_extra_metrics(self, metrics: Mapping[str, float]) -> None:
        """Store additional metrics (e.g., perception confidence) for next update."""

        for key, value in metrics.items():
            try:
                self._extra_metrics[key] = float(value)
            except Exception:
                continue

    def _evaluate_feedback(self, metrics: Mapping[str, Any]) -> None:
        feedback = self._feedback_evaluator.evaluate(metrics)
        if not feedback:
            return
        self._feedback_log.extend(feedback)
        if self._monitor is not None:
            try:
                self._monitor.log_snapshot({"feedback_count": len(feedback)})
            except Exception:
                pass

    def _apply_feedback(self) -> None:
        """Trigger self-improvement routines based on accumulated feedback."""

        if not self._feedback_log:
            return
        pending = list(self._feedback_log)
        self._feedback_log.clear()
        for fb in pending:
            module = fb.get("module")
            fb_type = fb.get("type")
            value = fb.get("value")
            if module == "decision" and self._meta_adjustment_provider:
                try:
                    reduce_explore = fb_type in {"low_reward", "low_reward_avg"}
                    self._meta_adjustment_provider(
                        {
                            "reason": fb_type,
                            "value": value,
                            "suggested": {"exploration": "decrease" if reduce_explore else "increase"},
                        }
                    )
                except Exception:
                    pass
            if module == "perception" and self._retrain_callback:
                try:
                    self._retrain_callback({"module": module, "reason": fb_type, "value": value})
                except Exception:
                    pass
            if module == "module_latency" and self._resource_optimizer:
                try:
                    self._resource_optimizer({"action": "reduce_load", "value": value})
                except Exception:
                    pass
            if module == "memory" and self._resource_optimizer:
                try:
                    self._resource_optimizer({"action": "optimize_memory", "value": value})
                except Exception:
                    pass

    def _apply_meta_learning(self, metrics: Mapping[str, Any]) -> None:
        """Send trend-based meta-adjustments without waiting for explicit feedback."""

        if self._meta_adjustment_provider is None or self._meta_tuner is None:
            return
        suggestion = self._meta_tuner.observe(metrics)
        if not suggestion:
            return
        try:
            self._meta_adjustment_provider(suggestion)
        except Exception:
            self._logger.debug("Meta-learning adjustment failed", exc_info=True)

    def _adapt_goals(self, metrics: Mapping[str, Any]) -> None:
        """Tighten internal targets based on achieved metrics."""

        if self._self_improvement is None:
            return
        if "perception_prediction_error" in metrics:
            self._self_improvement.ensure_goal(
                "perception_prediction_error", metrics["perception_prediction_error"], direction="decrease"
            )
        if "decision_success_rate" in metrics:
            self._self_improvement.ensure_goal("decision_success_rate", metrics["decision_success_rate"], direction="increase")
        if "decision_reward_avg" in metrics:
            self._self_improvement.ensure_goal("decision_reward_avg", metrics["decision_reward_avg"], direction="increase")
        if "memory_hit_rate" in metrics:
            self._self_improvement.ensure_goal("memory_hit_rate", metrics["memory_hit_rate"], direction="increase")
        achieved = self._self_improvement.observe_metrics(metrics)
        if achieved and self._monitor is not None:
            try:
                self._monitor.log_snapshot({"goals_achieved": len(achieved)})
            except Exception:
                pass

    def shutdown(self) -> None:
        """Reset mitigations before shutdown."""

        if self._mode == "low" and self._initial_big_brain:
            self._restore_high_fidelity(time.time(), 0.0, force=True)

    # ------------------------------------------------------------------
    def _handle_memory_pressure(self, now: float, avg_memory: float) -> None:
        if now - self._last_memory_compaction < self._memory_cooldown:
            return
        removed = 0
        try:
            if self._memory_router is not None:
                removed = int(
                    self._memory_router.shrink(
                        max_entries=256,
                        max_age=1800.0,
                        min_usage=1,
                    )
                    or 0
                )
        except Exception:  # pragma: no cover - defensive guard
            self._logger.debug("MemoryRouter shrink failed", exc_info=True)

        compressed = False
        try:
            if self._long_term_memory is not None:
                self._long_term_memory.compress()
                compressed = True
        except Exception:  # pragma: no cover - defensive guard
            self._logger.debug("Long-term memory compression failed", exc_info=True)

        self._last_memory_compaction = now
        if self._event_bus:
            self._event_bus.publish(
                "resource.adaptation.memory",
                {
                    "avg_memory": avg_memory,
                    "entries_pruned": removed,
                    "long_term_compressed": compressed,
                },
            )
        self._logger.info(
            "Memory pressure %.1f%% -> pruned %d items%s",
            avg_memory,
            removed,
            " and vacuumed" if compressed else "",
        )

    def _enter_low_fidelity(self, now: float, avg_cpu: float, backlog: int) -> None:
        if self._mode == "low" or now - self._last_mode_change < self._mode_cooldown:
            return
        try:
            setattr(self._config, "big_brain", False)
        except Exception:  # pragma: no cover - defensive guard
            self._logger.debug("Unable to adjust config.big_brain", exc_info=True)
        self._mode = "low"
        self._last_mode_change = now
        if self._event_bus:
            self._event_bus.publish(
                "resource.adaptation.mode",
                {"mode": "low", "avg_cpu": avg_cpu, "backlog": backlog},
            )
        self._logger.warning(
            "High CPU utilisation %.1f%% with backlog %d -> switching to fast LLM",
            avg_cpu,
            backlog,
        )

    def _restore_high_fidelity(
        self,
        now: float,
        avg_cpu: float,
        force: bool = False,
    ) -> None:
        if self._mode == "high":
            return
        if not force and now - self._last_mode_change < self._mode_cooldown:
            return
        try:
            setattr(self._config, "big_brain", self._initial_big_brain)
        except Exception:  # pragma: no cover - defensive guard
            self._logger.debug("Unable to restore config.big_brain", exc_info=True)
        self._mode = "high" if self._initial_big_brain else "low"
        self._last_mode_change = now
        if self._event_bus:
            self._event_bus.publish(
                "resource.adaptation.mode",
                {"mode": self._mode, "avg_cpu": avg_cpu, "backlog": 0},
            )
        self._logger.info(
            "CPU utilisation %.1f%% -> restoring %s fidelity",
            avg_cpu,
            "high" if self._initial_big_brain else "current",
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _default_reward(metrics: Mapping[str, Any]) -> float:
        reward = float(metrics.get("reward", 0.0) or 0.0)
        reward += float(metrics.get("success_rate", 0.0) or 0.0)
        reward -= 0.05 * float(metrics.get("avg_cpu", 0.0) or 0.0) / 100.0
        reward -= 0.05 * float(metrics.get("avg_memory", 0.0) or 0.0) / 100.0
        backlog = float(metrics.get("backlog", 0.0) or 0.0)
        reward -= 0.01 * backlog
        return reward


# ---------------------------------------------------------------------------
# Meta-parameter reinforcement learning
# ---------------------------------------------------------------------------


@dataclass
class MetaParameterSpec:
    """Description of a tunable meta-parameter."""

    name: str
    values: Sequence[Any]
    epsilon: float = 0.15
    alpha: float = 0.3


class EpsilonGreedyMetaPolicy:
    """Simple epsilon-greedy bandit for discrete meta-parameters."""

    def __init__(
        self,
        specs: Iterable[MetaParameterSpec],
        *,
        default_epsilon: float = 0.15,
        default_alpha: float = 0.3,
    ) -> None:
        self._specs: Dict[str, MetaParameterSpec] = {}
        self._estimates: Dict[str, list[float]] = {}
        self._counts: Dict[str, list[int]] = {}
        self._last_choice: Dict[str, int] = {}
        self._epsilon = default_epsilon
        self._alpha = default_alpha
        for spec in specs:
            if not spec.values:
                continue
            self._specs[spec.name] = spec
            self._estimates[spec.name] = [0.0 for _ in spec.values]
            self._counts[spec.name] = [0 for _ in spec.values]

    def apply(
        self,
        config: Any,
        metrics: Mapping[str, Any],
        *,
        rng: Optional[random.Random] = None,
    ) -> Dict[str, Any]:
        """Choose new parameter assignments and apply them to ``config``."""

        rng = rng or random
        assignments: Dict[str, Any] = {}
        for name, spec in self._specs.items():
            epsilon = spec.epsilon if spec.epsilon is not None else self._epsilon
            estimates = self._estimates[name]
            if rng.random() < epsilon:
                idx = rng.randrange(len(spec.values))
            else:
                max_val = max(estimates)
                best_indices = [i for i, val in enumerate(estimates) if val == max_val]
                idx = rng.choice(best_indices)
            value = spec.values[idx]
            try:
                setattr(config, name, value)
            except Exception:
                # Fall back to storing in dictionary-like configs
                if isinstance(config, dict):
                    config[name] = value
            assignments[name] = value
            self._last_choice[name] = idx
        return assignments

    def record_outcome(self, reward: float) -> None:
        """Update value estimates using the observed ``reward``."""

        for name, idx in list(self._last_choice.items()):
            estimates = self._estimates[name]
            counts = self._counts[name]
            alpha = self._specs[name].alpha if self._specs[name].alpha is not None else self._alpha
            counts[idx] += 1
            estimates[idx] += alpha * (reward - estimates[idx])


# ---------------------------------------------------------------------------
# Evolutionary architecture search
# ---------------------------------------------------------------------------


@dataclass
class ModuleAdapter:
    """Describe a module that can be toggled or grown via architecture genes."""

    name: str
    enable: Callable[[], Any]
    disable: Callable[[], Any]
    enabled_probe: Optional[Callable[[], bool]] = None
    scale_probe: Optional[Callable[[], float]] = None
    apply_scale: Optional[Callable[[float], Any]] = None
    min_scale: float = 0.1
    max_scale: float = 10.0


@dataclass
class ArchitectureHotloader:
    """Derive and apply architecture genomes to live modules."""

    runtime_config: Any | None = None
    brain_config: Any | None = None
    memory_manager: Any | None = None
    curiosity_state: Any | None = None
    policy_module: Any | None = None
    learning_modules: Sequence[Any] | None = None
    reflection_controller: Any | None = None
    _defaults: Dict[str, float] = field(default_factory=dict)
    last_applied: Dict[str, float] | None = None
    module_adapters: Sequence["ModuleAdapter"] | None = None

    def derive_baseline(self) -> Dict[str, float]:
        """Capture current runtime knobs as a GA-friendly genome."""

        if self._defaults:
            return dict(self._defaults)

        policy_obj = self.policy_module
        nested_policy = getattr(policy_obj, "policy", None) if policy_obj is not None else None
        if nested_policy is not None:
            policy_obj = nested_policy

        policy_variant = 1.0 if getattr(self.brain_config, "prefer_reinforcement_planner", False) else 0.0
        if policy_obj is not None:
            policy_name = type(policy_obj).__name__
            if "Bandit" in policy_name:
                policy_variant = 2.0
            elif "Reinforcement" in policy_name:
                policy_variant = 1.0

        planner_min_steps = 4.0
        planner = getattr(policy_obj, "planner", None) if policy_obj is not None else None
        if planner is not None:
            try:
                planner_min_steps = float(int(getattr(planner, "min_steps", 4) or 4))
            except Exception:
                planner_min_steps = 4.0

        replay_buffer = getattr(policy_obj, "_experience_buffer", None) if policy_obj is not None else None
        replay_buffer_size = float(getattr(replay_buffer, "maxlen", 256) or 256)
        replay_batch_size = float(getattr(policy_obj, "_replay_batch_size", 16) or 16) if policy_obj is not None else 16.0
        replay_iterations = float(getattr(policy_obj, "_replay_iterations", 1) or 1) if policy_obj is not None else 1.0

        cfg = getattr(policy_obj, "config", None) if policy_obj is not None else None
        hidden_dim = float(getattr(cfg, "hidden_dim", 128) or 128) if cfg is not None else 128.0
        num_layers = float(getattr(cfg, "num_layers", 2) or 2) if cfg is not None else 2.0

        defaults: Dict[str, float] = {
            "big_brain_flag": 1.0 if getattr(self.runtime_config, "big_brain", True) else 0.0,
            "parallel_branches": float(getattr(self.runtime_config, "parallel_branches", 1.0) or 1.0),
            "planner_structured_flag": 1.0
            if getattr(self.brain_config, "prefer_structured_planner", True)
            else 0.0,
            "planner_reinforcement_flag": 1.0
            if getattr(self.brain_config, "prefer_reinforcement_planner", False)
            else 0.0,
            "module_self_learning_flag": 1.0
            if getattr(self.brain_config, "enable_self_learning", True)
            else 0.0,
            "module_curiosity_feedback_flag": 1.0
            if getattr(self.brain_config, "enable_curiosity_feedback", True)
            else 0.0,
            "module_metrics_flag": 1.0
            if getattr(self.brain_config, "metrics_enabled", True)
            else 0.0,
            "memory_short_term_limit": float(getattr(self.memory_manager, "_short_term_limit", 25.0)),
            "memory_working_limit": float(getattr(self.memory_manager, "_working_limit", 50.0)),
            "memory_summary_batch_size": float(
                getattr(self.memory_manager, "_summary_batch_size", 5.0)
            ),
            "memory_summary_rate_limit": float(
                getattr(self.memory_manager, "_summary_rate_limit", 900.0)
            ),
            "curiosity_drive_floor": float(getattr(self.curiosity_state, "drive", 0.4)),
            "curiosity_novelty_preference": float(
                getattr(self.curiosity_state, "novelty_preference", 0.5)
            ),
            "curiosity_fatigue_ceiling": float(getattr(self.curiosity_state, "fatigue", 0.1)),
            "policy_learning_rate": float(getattr(self.policy_module, "learning_rate", 0.08)),
            "policy_exploration_rate": float(getattr(self.policy_module, "exploration", 0.12)),
            "cognitive_policy_variant": float(policy_variant),
            "planner_min_steps": float(planner_min_steps),
            "policy_replay_buffer_size": float(replay_buffer_size),
            "policy_replay_batch_size": float(replay_batch_size),
            "policy_replay_iterations": float(replay_iterations),
            "policy_hidden_dim": float(hidden_dim),
            "policy_num_layers": float(num_layers),
            "reflection_interval_hours": float(
                getattr(
                    self.reflection_controller,
                    "reflection_interval_hours",
                    getattr(self.reflection_controller, "interval_hours", 24.0),
                )
            ),
        }
        for adapter in self.module_adapters or []:
            key = f"module_{adapter.name}_flag"
            current = 1.0
            if adapter.enabled_probe is not None:
                try:
                    current = 1.0 if adapter.enabled_probe() else 0.0
                except Exception:
                    current = 1.0
            defaults[key] = current
            if adapter.scale_probe is not None:
                try:
                    scale = float(adapter.scale_probe())
                except Exception:
                    scale = 1.0
            else:
                scale = 1.0
            defaults[f"module_{adapter.name}_scale"] = scale
        self._defaults = defaults
        return dict(defaults)

    # ------------------------------------------------------------------ #
    def _normalise(self, arch: Mapping[str, float]) -> Dict[str, float]:
        defaults = self._defaults or self.derive_baseline()
        normalised = dict(defaults)
        normalised.update(arch or {})

        normalised["big_brain_flag"] = 1.0 if float(normalised.get("big_brain_flag", 1.0)) >= 0.5 else 0.0
        branches = float(normalised.get("parallel_branches", defaults.get("parallel_branches", 1.0)))
        normalised["parallel_branches"] = max(1.0, branches)

        short_limit = max(1.0, float(normalised.get("memory_short_term_limit", defaults["memory_short_term_limit"])))
        working_limit = max(short_limit, float(normalised.get("memory_working_limit", defaults["memory_working_limit"])))
        normalised["memory_short_term_limit"] = short_limit
        normalised["memory_working_limit"] = working_limit

        for flag in (
            "planner_structured_flag",
            "planner_reinforcement_flag",
            "module_self_learning_flag",
            "module_curiosity_feedback_flag",
            "module_metrics_flag",
        ):
            raw = float(normalised.get(flag, defaults.get(flag, 0.0)))
            normalised[flag] = 1.0 if raw >= 0.5 else 0.0
        for adapter in self.module_adapters or []:
            key = f"module_{adapter.name}_flag"
            raw = float(normalised.get(key, defaults.get(key, 1.0)))
            normalised[key] = 1.0 if raw >= 0.5 else 0.0
            scale_key = f"module_{adapter.name}_scale"
            scale = float(normalised.get(scale_key, defaults.get(scale_key, 1.0)))
            scale = max(adapter.min_scale, min(adapter.max_scale, scale))
            normalised[scale_key] = scale

        normalised["policy_learning_rate"] = max(
            1e-5, min(1.0, float(normalised.get("policy_learning_rate", defaults["policy_learning_rate"])))
        )
        normalised["policy_exploration_rate"] = max(
            0.0, min(1.0, float(normalised.get("policy_exploration_rate", defaults["policy_exploration_rate"])))
        )
        normalised["memory_summary_batch_size"] = float(
            max(1, int(round(normalised.get("memory_summary_batch_size", defaults["memory_summary_batch_size"]))))
        )
        normalised["memory_summary_rate_limit"] = max(
            60.0, float(normalised.get("memory_summary_rate_limit", defaults["memory_summary_rate_limit"]))
        )
        normalised["reflection_interval_hours"] = max(
            0.25, float(normalised.get("reflection_interval_hours", defaults["reflection_interval_hours"]))
        )
        for key in (
            "curiosity_drive_floor",
            "curiosity_novelty_preference",
            "curiosity_fatigue_ceiling",
        ):
            value = float(normalised.get(key, defaults.get(key, 0.0)))
            normalised[key] = max(0.0, min(1.0, value))

        normalised["cognitive_policy_variant"] = float(
            max(0, min(2, int(round(float(normalised.get("cognitive_policy_variant", 0.0))))))
        )
        normalised["planner_min_steps"] = float(
            max(1, min(16, int(round(float(normalised.get("planner_min_steps", 4.0))))))
        )
        normalised["policy_replay_buffer_size"] = float(
            max(32, min(4096, int(round(float(normalised.get("policy_replay_buffer_size", 256.0))))))
        )
        normalised["policy_replay_batch_size"] = float(
            max(1, min(256, int(round(float(normalised.get("policy_replay_batch_size", 16.0))))))
        )
        normalised["policy_replay_iterations"] = float(
            max(1, min(12, int(round(float(normalised.get("policy_replay_iterations", 1.0))))))
        )
        normalised["policy_hidden_dim"] = float(
            max(8, min(2048, int(round(float(normalised.get("policy_hidden_dim", 128.0))))))
        )
        normalised["policy_num_layers"] = float(
            max(1, min(8, int(round(float(normalised.get("policy_num_layers", 2.0))))))
        )
        return normalised

    # ------------------------------------------------------------------ #
    def apply(self, arch: Mapping[str, float]) -> None:
        """Hot-load an architecture into the attached runtime modules."""

        normalised = self._normalise(arch)
        self.last_applied = dict(normalised)

        if self.runtime_config is not None and hasattr(self.runtime_config, "big_brain"):
            try:
                self.runtime_config.big_brain = bool(normalised.get("big_brain_flag", 1.0) >= 0.5)
            except Exception:  # pragma: no cover - defensive
                pass
        if self.runtime_config is not None and hasattr(self.runtime_config, "parallel_branches"):
            try:
                self.runtime_config.parallel_branches = float(normalised.get("parallel_branches", 1.0))
            except Exception:  # pragma: no cover
                pass

        if self.memory_manager is not None:
            try:
                self.memory_manager._short_term_limit = int(round(normalised["memory_short_term_limit"]))
                self.memory_manager._working_limit = int(round(normalised["memory_working_limit"]))
                if hasattr(self.memory_manager, "_summary_batch_size"):
                    self.memory_manager._summary_batch_size = int(
                        round(normalised["memory_summary_batch_size"])
                    )
                if hasattr(self.memory_manager, "_summary_rate_limit"):
                    self.memory_manager._summary_rate_limit = float(
                        normalised["memory_summary_rate_limit"]
                    )
            except Exception:  # pragma: no cover - defensive
                pass

        if self.curiosity_state is not None:
            for attr, key in (
                ("drive", "curiosity_drive_floor"),
                ("novelty_preference", "curiosity_novelty_preference"),
                ("fatigue", "curiosity_fatigue_ceiling"),
            ):
                try:
                    setattr(self.curiosity_state, attr, float(normalised.get(key, getattr(self.curiosity_state, attr, 0.0))))
                except Exception:  # pragma: no cover
                    pass

        if self.brain_config is not None:
            try:
                self.brain_config.prefer_structured_planner = bool(
                    normalised.get("planner_structured_flag", 1.0) >= 0.5
                )
                self.brain_config.prefer_reinforcement_planner = bool(
                    normalised.get("planner_reinforcement_flag", 0.0) >= 0.5
                )
                if "module_self_learning_flag" in normalised:
                    self.brain_config.enable_self_learning = bool(
                        normalised.get("module_self_learning_flag", 1.0) >= 0.5
                    )
                if "module_curiosity_feedback_flag" in normalised:
                    self.brain_config.enable_curiosity_feedback = bool(
                        normalised.get("module_curiosity_feedback_flag", 1.0) >= 0.5
                    )
                if "module_metrics_flag" in normalised:
                    self.brain_config.metrics_enabled = bool(
                        normalised.get("module_metrics_flag", 1.0) >= 0.5
                    )
            except Exception:  # pragma: no cover - defensive
                pass

        policy_variant = int(round(float(normalised.get("cognitive_policy_variant", 0.0))))
        planner_min_steps = int(round(float(normalised.get("planner_min_steps", 4.0))))
        replay_buffer_size = int(round(float(normalised.get("policy_replay_buffer_size", 256.0))))
        replay_batch_size = int(round(float(normalised.get("policy_replay_batch_size", 16.0))))
        replay_iterations = int(round(float(normalised.get("policy_replay_iterations", 1.0))))
        hidden_dim = int(round(float(normalised.get("policy_hidden_dim", 128.0))))
        num_layers = int(round(float(normalised.get("policy_num_layers", 2.0))))

        if (
            self.policy_module is not None
            and hasattr(self.policy_module, "set_policy")
            and hasattr(self.policy_module, "policy")
        ):
            try:  # pragma: no cover - optional brain policy stack
                from modules.brain.whole_brain_policy import (
                    BanditCognitivePolicy,
                    ProductionCognitivePolicy,
                    ReinforcementCognitivePolicy,
                    StructuredPlanner,
                )
            except Exception:  # pragma: no cover
                BanditCognitivePolicy = None  # type: ignore[assignment]
                ProductionCognitivePolicy = None  # type: ignore[assignment]
                ReinforcementCognitivePolicy = None  # type: ignore[assignment]
                StructuredPlanner = None  # type: ignore[assignment]

            current_policy = getattr(self.policy_module, "policy", None)
            if (
                current_policy is not None
                and StructuredPlanner is not None
                and ProductionCognitivePolicy is not None
                and ReinforcementCognitivePolicy is not None
            ):
                planner = getattr(current_policy, "planner", None)
                if planner is None or not hasattr(planner, "min_steps"):
                    planner = StructuredPlanner(min_steps=planner_min_steps)
                else:
                    try:
                        planner.min_steps = max(1, int(planner_min_steps))
                    except Exception:
                        pass

                desired_policy: Any | None = None
                if policy_variant == 2 and BanditCognitivePolicy is not None:
                    if not isinstance(current_policy, BanditCognitivePolicy):
                        desired_policy = BanditCognitivePolicy(
                            exploration=float(normalised.get("policy_exploration_rate", 0.12)),
                            planner=planner,
                            fallback=ProductionCognitivePolicy(planner=planner),
                        )
                elif policy_variant == 1:
                    if not isinstance(current_policy, ReinforcementCognitivePolicy):
                        desired_policy = ReinforcementCognitivePolicy(
                            learning_rate=float(normalised.get("policy_learning_rate", 0.08)),
                            exploration=float(normalised.get("policy_exploration_rate", 0.12)),
                            planner=planner,
                            replay_buffer_size=replay_buffer_size,
                            replay_batch_size=replay_batch_size,
                            replay_iterations=replay_iterations,
                        )
                else:
                    if not isinstance(current_policy, ProductionCognitivePolicy):
                        desired_policy = ProductionCognitivePolicy(planner=planner)

                if desired_policy is not None:
                    try:
                        self.policy_module.set_policy(desired_policy)
                    except Exception:
                        pass

        targets: list[Any] = [self.policy_module, *(self.learning_modules or [])]
        expanded_targets: list[Any] = []
        for target in targets:
            if target is None:
                continue
            expanded_targets.append(target)
            nested = getattr(target, "policy", None)
            if nested is not None:
                expanded_targets.append(nested)
        targets = expanded_targets
        for target in targets:
            if target is None:
                continue
            if hasattr(target, "learning_rate"):
                try:
                    target.learning_rate = float(normalised["policy_learning_rate"])
                except Exception:  # pragma: no cover
                    pass
            if hasattr(target, "exploration"):
                try:
                    target.exploration = float(normalised["policy_exploration_rate"])
                except Exception:  # pragma: no cover
                    pass
            planner = getattr(target, "planner", None)
            if planner is not None and hasattr(planner, "min_steps"):
                try:
                    planner.min_steps = max(1, int(planner_min_steps))
                except Exception:  # pragma: no cover
                    pass
            if hasattr(target, "_experience_buffer"):
                buffer = getattr(target, "_experience_buffer", None)
                maxlen = getattr(buffer, "maxlen", None) if buffer is not None else None
                if isinstance(maxlen, int) and maxlen != replay_buffer_size:
                    items = list(buffer) if buffer is not None else []
                    trimmed = items[-replay_buffer_size:] if replay_buffer_size > 0 else []
                    try:
                        target._experience_buffer = deque(trimmed, maxlen=max(1, replay_buffer_size))
                    except Exception:  # pragma: no cover
                        pass
            if hasattr(target, "_replay_batch_size"):
                try:
                    target._replay_batch_size = max(1, int(replay_batch_size))
                except Exception:  # pragma: no cover
                    pass
            if hasattr(target, "_replay_iterations"):
                try:
                    target._replay_iterations = max(1, int(replay_iterations))
                except Exception:  # pragma: no cover
                    pass
            if hasattr(target, "update_architecture"):
                try:
                    target.update_architecture(hidden_dim=hidden_dim, num_layers=num_layers)
                except TypeError:  # pragma: no cover
                    pass
                except Exception:  # pragma: no cover
                    pass

        for adapter in self.module_adapters or []:
            key = f"module_{adapter.name}_flag"
            enabled = bool(normalised.get(key, 1.0) >= 0.5)
            scale_key = f"module_{adapter.name}_scale"
            scale_value = float(normalised.get(scale_key, 1.0))
            current_enabled = None
            if adapter.enabled_probe is not None:
                try:
                    current_enabled = bool(adapter.enabled_probe())
                except Exception:
                    current_enabled = None
            try:
                if enabled and (current_enabled is None or current_enabled is False):
                    adapter.enable()
                elif not enabled and (current_enabled is None or current_enabled is True):
                    adapter.disable()
            except Exception:  # pragma: no cover - best effort application
                pass
            if enabled and adapter.apply_scale is not None:
                try:
                    adapter.apply_scale(scale_value)
                except Exception:  # pragma: no cover - best effort
                    pass

        if self.reflection_controller is not None:
            if hasattr(self.reflection_controller, "reflection_interval_hours"):
                try:
                    self.reflection_controller.reflection_interval_hours = float(
                        normalised["reflection_interval_hours"]
                    )
                except Exception:  # pragma: no cover
                    pass
            elif hasattr(self.reflection_controller, "interval_hours"):
                try:
                    self.reflection_controller.interval_hours = float(
                        normalised["reflection_interval_hours"]
                    )
                except Exception:  # pragma: no cover
                    pass


def _default_architecture_reward(
    architecture: Mapping[str, float],
    samples: Sequence[Mapping[str, Any]],
) -> float:
    if not samples:
        return 0.0
    reward_terms = []
    latency_penalties = []
    resource_penalties = []
    module_latency_penalties = []
    for sample in samples:
        reward_terms.append(float(sample.get("reward", sample.get("success_rate", 0.0)) or 0.0))
        reward_terms.append(float(sample.get("success_rate", 0.0) or 0.0))
        reward_terms.append(0.1 * float(sample.get("throughput", sample.get("avg_throughput", 0.0)) or 0.0))
        latency = float(sample.get("latency", sample.get("avg_latency", 0.0)) or 0.0)
        if latency:
            latency_penalties.append(latency)
        cpu = float(sample.get("avg_cpu", 0.0) or 0.0)
        mem = float(sample.get("avg_memory", 0.0) or 0.0)
        backlog = float(sample.get("backlog", 0.0) or 0.0)
        resource_penalties.append(0.01 * cpu + 0.01 * mem + 0.05 * backlog)
        mod_lat = float(sample.get("module_latency_max", 0.0) or 0.0)
        if mod_lat:
            module_latency_penalties.append(mod_lat)
    base = sum(reward_terms) / max(len(reward_terms), 1)
    latency_penalty = sum(latency_penalties) / max(len(latency_penalties), 1) if latency_penalties else 0.0
    resource_penalty = sum(resource_penalties) / max(len(resource_penalties), 1) if resource_penalties else 0.0
    module_penalty = sum(module_latency_penalties) / max(len(module_latency_penalties), 1) if module_latency_penalties else 0.0
    penalty = (
        0.01 * sum(abs(float(val)) for val in architecture.values())
        + 0.1 * latency_penalty
        + resource_penalty
        + 0.05 * module_penalty
    )
    return base - penalty


@dataclass
class HybridArchitectureManager:
    """Coordinate evolutionary search over architectural hyper-parameters."""

    initial_architecture: Dict[str, float]
    evaluator: Callable[[Mapping[str, float], Sequence[Mapping[str, Any]]], float] = _default_architecture_reward
    apply_callback: Optional[Callable[[Dict[str, float]], None]] = None
    ga_config: GAConfig | None = None
    history: Deque[Mapping[str, Any]] = field(default_factory=lambda: deque(maxlen=10))
    min_improvement: float = 0.02
    cooldown_steps: int = 5
    seed: Optional[int] = None
    hotloader: Optional["ArchitectureHotloader"] = None
    pso_bounds: Optional[Mapping[str, Tuple[float, float]]] = None
    pso_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.hotloader is not None:
            baseline = self.hotloader.derive_baseline()
            merged = baseline.copy()
            merged.update(self.initial_architecture or {})
            self.initial_architecture = merged
            if self.apply_callback is None:
                self.apply_callback = self.hotloader.apply
        if self.ga_config is None:
            self.ga_config = GAConfig(population_size=12, generations=4, mutation_sigma=0.2)
        self._ga = GeneticAlgorithm(self._score_candidate, self.ga_config, seed=self.seed)
        self._current_arch = self.initial_architecture.copy()
        self._step = 0
        self._last_evolution_step = -self.cooldown_steps
        self._scores: Deque[Tuple[Dict[str, float], float]] = deque(maxlen=20)
        if self.hotloader is not None and self.apply_callback is not None:
            try:
                self.apply_callback(self._current_arch.copy())
            except Exception:  # pragma: no cover - defensive best-effort
                pass

    @classmethod
    def from_runtime(
        cls,
        *,
        runtime_config: Any | None = None,
        brain_config: Any | None = None,
        memory_manager: Any | None = None,
        curiosity_state: Any | None = None,
        policy_module: Any | None = None,
        learning_modules: Sequence[Any] | None = None,
        reflection_controller: Any | None = None,
        initial_overrides: Mapping[str, float] | None = None,
        ga_config: GAConfig | None = None,
        history: Deque[Mapping[str, Any]] | None = None,
        min_improvement: float = 0.02,
        cooldown_steps: int = 5,
        seed: int | None = None,
        evaluator: Callable[[Mapping[str, float], Sequence[Mapping[str, Any]]], float] | None = None,
        module_adapters: Sequence["ModuleAdapter"] | None = None,
        pso_bounds: Mapping[str, Tuple[float, float]] | None = None,
        pso_config: Optional[Dict[str, Any]] = None,
    ) -> "HybridArchitectureManager":
        hotloader = ArchitectureHotloader(
            runtime_config=runtime_config,
            brain_config=brain_config,
            memory_manager=memory_manager,
            curiosity_state=curiosity_state,
            policy_module=policy_module,
            learning_modules=learning_modules,
            reflection_controller=reflection_controller,
            module_adapters=module_adapters,
        )
        baseline = hotloader.derive_baseline()
        initial_arch = baseline.copy()
        if initial_overrides:
            initial_arch.update(initial_overrides)
        return cls(
            initial_architecture=initial_arch,
            evaluator=evaluator or _default_architecture_reward,
            apply_callback=None,
            ga_config=ga_config,
            history=history or deque(maxlen=10),
            min_improvement=min_improvement,
            cooldown_steps=cooldown_steps,
            seed=seed,
            hotloader=hotloader,
            pso_bounds=pso_bounds,
            pso_config=pso_config or {},
        )

    def observe(self, metrics: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        self.record(metrics)
        if self._step - self._last_evolution_step < self.cooldown_steps:
            return None
        if len(self.history) < self.history.maxlen:
            return None

        baseline = self._score_candidate(self._current_arch)
        best_arch, best_score, _ = self._ga.evolve(self._current_arch)
        best_arch, best_score = self._maybe_pso_refine(best_arch, best_score)
        if best_score > baseline + self.min_improvement:
            self._current_arch = best_arch.copy()
            self._last_evolution_step = self._step
            self._scores.append((best_arch.copy(), best_score))
            if self.apply_callback is not None:
                try:
                    self.apply_callback(best_arch.copy())
                except Exception:
                    pass
            return {"architecture": best_arch.copy(), "score": best_score}
        return None

    def record(self, metrics: Mapping[str, Any]) -> None:
        """Record metrics without forcing an evolution step."""

        self.history.append(dict(metrics))
        self._step += 1

    def current_architecture(self) -> Dict[str, float]:
        return self._current_arch.copy()

    def _score_candidate(self, candidate: Dict[str, float]) -> float:
        return self.evaluator(candidate, list(self.history))

    # ------------------------------------------------------------------ #
    def _maybe_pso_refine(self, arch: Dict[str, float], score: float) -> Tuple[Dict[str, float], float]:
        """Optionally refine continuous hyperparameters using PSO."""

        if pso is None or not self.pso_bounds:
            return arch, score

        keys = [k for k in self.pso_bounds if k in arch]
        if not keys:
            return arch, score

        bounds = [tuple(map(float, self.pso_bounds[k])) for k in keys]
        base_arch = arch.copy()

        def _objective(vector: "np.ndarray") -> float:  # type: ignore[name-defined]
            candidate = base_arch.copy()
            for idx, key in enumerate(keys):
                candidate[key] = float(vector[idx])
            # Negative because GA uses higher-is-better fitness.
            return -self._score_candidate(candidate)

        try:
            result = pso(
                _objective,
                bounds=bounds,
                **{k: v for k, v in self.pso_config.items()},
            )
        except Exception:  # pragma: no cover - PSO optional
            return arch, score

        try:
            refined_score = -float(result.value)
        except Exception:
            return arch, score

        if refined_score <= score:
            return arch, score

        refined_arch = base_arch.copy()
        try:
            for idx, key in enumerate(keys):
                refined_arch[key] = float(result.position[idx])
        except Exception:
            return arch, score
        return refined_arch, refined_score


# ---------------------------------------------------------------------------
# Evolutionary self-improvement orchestration
# ---------------------------------------------------------------------------


@dataclass
class EvolutionCandidate:
    """Description of a structural or behavioural modification."""

    name: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None


@dataclass(frozen=True)
class KnowledgeConstraint:
    """Constraint guarding knowledge and safety boundaries."""

    name: str
    validator: Callable[[Mapping[str, Any]], bool]
    message: str

    def check(self, state: Mapping[str, Any]) -> bool:
        return self.validator(state)


class EvolutionConstraintViolation(RuntimeError):
    """Raised when a candidate violates knowledge or safety rules."""


class EvolutionGuard:
    """Evaluate candidates against configured safety constraints."""

    def __init__(self, constraints: Iterable[KnowledgeConstraint] | None = None) -> None:
        self._constraints = list(constraints or [])

    def allows(self, candidate: EvolutionCandidate) -> bool:
        state = {**candidate.payload, **candidate.metadata}
        for constraint in self._constraints:
            if not constraint.check(state):
                raise EvolutionConstraintViolation(
                    f"{constraint.name}: {constraint.message}"
                )
        return True


def immutable_components_constraint(components: Sequence[str]) -> KnowledgeConstraint:
    protected = {comp for comp in components}

    def _validator(state: Mapping[str, Any]) -> bool:
        touched = set(map(str, state.get("modified_components", [])))
        return not (protected & touched)

    return KnowledgeConstraint(
        name="immutable_components",
        validator=_validator,
        message=f"immutable components cannot be modified ({', '.join(sorted(protected))})",
    )


class PytestRegressionRunner:
    """Execute regression suites to validate evolved configurations."""

    def __init__(
        self,
        paths: Sequence[str] | None = None,
        *,
        timeout: Optional[float] = None,
    ) -> None:
        self.paths = list(paths or ["tests/test_new_modules.py"])
        self.timeout = timeout

    def __call__(self) -> bool:
        if not self.paths:
            return True
        cmd = [sys.executable, "-m", "pytest", *self.paths]
        result = subprocess.run(cmd, check=False, timeout=self.timeout)
        return result.returncode == 0


class SelfEvolutionLoop:
    """Coordinate observe -> generate -> evaluate -> select -> replace cycles."""

    def __init__(
        self,
        *,
        observer: Callable[[], Mapping[str, Any]],
        generator: Callable[[Mapping[str, Any]], Iterable[EvolutionCandidate]],
        evaluator: Callable[[EvolutionCandidate, Mapping[str, Any]], float],
        replacer: Callable[[EvolutionCandidate, Mapping[str, Any]], bool],
        guard: Optional[EvolutionGuard] = None,
        selector: Optional[
            Callable[[Sequence[EvolutionCandidate], Optional[float]], Optional[EvolutionCandidate]]
        ] = None,
        regression_runner: Optional[Callable[[], bool]] = None,
        min_improvement: float = 0.0,
        logger: Optional[logging.Logger] = None,
        diversity_archive: Optional["DiversityArchive"] = None,
    ) -> None:
        self._observer = observer
        self._generator = generator
        self._evaluator = evaluator
        self._replacer = replacer
        self._guard = guard
        self._selector = selector or self._default_selector
        self._regression_runner = regression_runner
        self._min_improvement = min_improvement
        self._logger = logger or logging.getLogger(__name__)
        self._last_score: Optional[float] = None
        self._diversity_archive = diversity_archive

    def run_cycle(self) -> Optional[EvolutionCandidate]:
        metrics = self._observer()
        candidates = []
        for candidate in self._generator(metrics):
            try:
                if self._guard:
                    self._guard.allows(candidate)
            except EvolutionConstraintViolation as exc:
                self._logger.warning("Candidate %s rejected: %s", candidate.name, exc)
                continue
            score = self._evaluator(candidate, metrics)
            candidate.score = score
            candidates.append(candidate)

        chosen = self._selector(candidates, self._last_score)
        if not chosen or chosen.score is None:
            return None

        if self._last_score is not None and chosen.score < self._last_score + self._min_improvement:
            return None

        applied = False
        try:
            applied = self._replacer(chosen, metrics)
        except Exception:  # pragma: no cover - defensive wrapper
            self._logger.exception("Failed to apply candidate %s", chosen.name)
            return None
        if not applied:
            return None

        if self._regression_runner is not None:
            if not self._regression_runner():
                self._logger.warning("Regression suite rejected candidate %s", chosen.name)
                return None

        self._last_score = chosen.score
        if self._diversity_archive is not None:
            try:
                self._diversity_archive.register(chosen)
            except Exception:  # pragma: no cover - archive issues should not break evolution
                self._logger.debug("Failed to register candidate in diversity archive", exc_info=True)
        self._logger.info("Evolution cycle accepted candidate %s (score %.4f)", chosen.name, chosen.score)
        return chosen

    @staticmethod
    def _default_selector(
        candidates: Sequence[EvolutionCandidate],
        last_score: Optional[float],
    ) -> Optional[EvolutionCandidate]:
        if not candidates:
            return None
        return max(candidates, key=lambda cand: cand.score or float("-inf"))


class EvolutionDaemon:
    """Background thread continuously executing self-evolution cycles."""

    def __init__(
        self,
        loop: SelfEvolutionLoop,
        *,
        interval: float = 120.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._loop = loop
        self._interval = max(1.0, interval)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._logger = logger or logging.getLogger(__name__)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        def _runner() -> None:
            while not self._stop_event.is_set():
                try:
                    self._loop.run_cycle()
                except Exception:  # pragma: no cover - defensive
                    self._logger.exception("Evolution cycle failed")
                self._stop_event.wait(self._interval)

        self._thread = threading.Thread(target=_runner, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._interval + 5.0)


@dataclass
class DiversityArchive:
    """Maintain a set of high-performing, stylistically diverse candidates."""

    max_size: int = 3
    distance_fn: Callable[[EvolutionCandidate, EvolutionCandidate], float] | None = None
    min_distance: float = 0.2
    _entries: Deque[EvolutionCandidate] = field(default_factory=deque)

    def register(self, candidate: EvolutionCandidate) -> None:
        entries = list(self._entries)
        if self.distance_fn:
            for existing in entries:
                if self.distance_fn(existing, candidate) < self.min_distance:
                    if (existing.score or 0.0) >= (candidate.score or 0.0):
                        return
                    entries.remove(existing)
                    break
        entries.append(candidate)
        entries.sort(key=lambda cand: cand.score or float("-inf"), reverse=True)
        trimmed = entries[: self.max_size]
        self._entries = deque(trimmed, maxlen=self.max_size)

    def entries(self) -> Sequence[EvolutionCandidate]:
        return list(self._entries)


__all__ = [
    "AdaptiveResourceController",
    "MetaParameterSpec",
    "EpsilonGreedyMetaPolicy",
    "HybridArchitectureManager",
    "EvolutionCandidate",
    "KnowledgeConstraint",
    "EvolutionGuard",
    "EvolutionConstraintViolation",
    "immutable_components_constraint",
    "SelfEvolutionLoop",
    "PytestRegressionRunner",
    "EvolutionDaemon",
    "DiversityArchive",
    "ArchitectureHotloader",
    "ModuleAdapter",
]
