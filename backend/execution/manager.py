"""Agent lifecycle manager.

This component watches for blueprint changes and ensures that running agents are
reloaded or spawned to reflect the latest blueprints. Reload results are
published on the global event bus for observability.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum
from collections import deque
import json
import gc
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore[assignment]
import gc

from agent_factory import create_agent_from_blueprint
from events import EventBus
from backend.monitoring import ResourceScheduler, SystemMetricsCollector, global_workspace
from common import AutoGPTException, log_and_format_exception
from org_charter.watchdog import BlueprintWatcher
from third_party.autogpt.autogpt.config import Config
from third_party.autogpt.autogpt.core.resource.model_providers import ChatModelProvider
from third_party.autogpt.autogpt.file_storage.base import FileStorage
from third_party.autogpt.autogpt.agents.agent import Agent
from .scheduler import Scheduler
from .planner import Planner
from .goal_generator import GoalGenerator
from .adaptive_controller import AdaptiveResourceController, HybridArchitectureManager, GAConfig, ModuleAdapter
from .task_manager import TaskHandle, TaskManager, TaskPriority
from world_model import WorldModel
from self_model import SelfModel
from capability.runtime_loader import RuntimeModuleManager
from knowledge import KnowledgeConsolidator, MemoryRouter
from modules.knowledge import KnowledgeUpdatePipeline, RuntimeKnowledgeImporter
from modules.environment.simulator import GridWorldEnvironment
from modules.environment.loop import ActionPerceptionLoop
from capability import (
    refresh_skills_from_directory,
    get_skill_registry,
)
from backend.monitoring import record_memory_hit, get_memory_hits
from .online_updates import apply_online_model_updates
from .imitation_updates import apply_online_imitation_updates
from .learning_manager import LearningManager
from .learning_safety import LearningSafetyGuard
from .self_debug_manager import SelfDebugManager
from .self_correction_manager import SelfCorrectionManager
from .self_diagnoser import SelfDiagnoser
from .automl_manager import AutoMLManager
from .code_self_repair import CodeSelfRepairManager
from .task_submission_scheduler import TaskSubmissionScheduler
try:  # Optional module acquisition (discover -> suggest -> install)
    from .module_acquisition import ModuleAcquisitionManager
except Exception:  # pragma: no cover - optional dependency
    ModuleAcquisitionManager = None  # type: ignore[assignment]
try:  # Optional high-level upgrade decision engine
    from .upgrade_decision_engine import UpgradeDecisionEngine
except Exception:  # pragma: no cover - optional dependency
    UpgradeDecisionEngine = None  # type: ignore[assignment]
try:  # Optional module lifecycle manager (track -> prune suggestions)
    from .module_lifecycle_manager import ModuleLifecycleManager
except Exception:  # pragma: no cover - optional dependency
    ModuleLifecycleManager = None  # type: ignore[assignment]
try:  # Optional failure-aware recovery manager (reload unstable modules)
    from .fault_recovery_manager import FaultRecoveryManager
except Exception:  # pragma: no cover - optional dependency
    FaultRecoveryManager = None  # type: ignore[assignment]
try:  # Optional event-driven scheduler control plane
    from .scheduler_control_manager import SchedulerControlManager
except Exception:  # pragma: no cover - optional dependency
    SchedulerControlManager = None  # type: ignore[assignment]
from .conductor import AgentConductor
try:  # Optional self-supervised predictor for world-model style forecasting
    from BrainSimulationSystem.learning.self_supervised import (
        SelfSupervisedConfig,
        SelfSupervisedPredictor,
    )
except Exception:  # pragma: no cover - optional dependency
    SelfSupervisedConfig = None  # type: ignore[assignment]
    SelfSupervisedPredictor = None  # type: ignore[assignment]
try:
    from backend.execution.self_improvement import SelfImprovementManager
except Exception:  # pragma: no cover - optional
    SelfImprovementManager = None  # type: ignore[assignment]
try:  # Optional cross-domain benchmark
    from benchmarks.run_cross_domain import CrossDomainBenchmark
except Exception:  # pragma: no cover - optional
    CrossDomainBenchmark = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


_CAPABILITY_TAG_RE = re.compile(r"\[(?:capability|capabilities):([^\]]+)\]", re.IGNORECASE)
_MODULE_TOKEN_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\\-]*$")


def _split_capability_tokens(value: str) -> list[str]:
    tokens = re.split(r"[,;\s]+", str(value or ""))
    return [token.strip() for token in tokens if token and token.strip()]


def _normalise_module_tokens(value: Any, *, lower: bool) -> list[str]:
    tokens: list[str] = []
    if value is None:
        return tokens
    if isinstance(value, str):
        tokens.extend(_split_capability_tokens(value))
    elif isinstance(value, dict):
        for entry in value.values():
            tokens.extend(_normalise_module_tokens(entry, lower=lower))
    elif isinstance(value, (list, tuple, set)):
        for entry in value:
            tokens.extend(_normalise_module_tokens(entry, lower=lower))

    cleaned: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        token = token.lower() if lower else token
        if not _MODULE_TOKEN_RE.match(token):
            continue
        if token in seen:
            continue
        cleaned.append(token)
        seen.add(token)
    return cleaned


def _extract_required_modules_from_plan_event(event: Dict[str, Any]) -> list[str]:
    required: list[str] = []
    seen: set[str] = set()

    def _extend(values: list[str]) -> None:
        for item in values:
            if item in seen:
                continue
            required.append(item)
            seen.add(item)

    for container in (event, event.get("metadata")):
        if not isinstance(container, dict):
            continue
        _extend(_normalise_module_tokens(container.get("required_modules") or container.get("modules"), lower=False))
        _extend(_normalise_module_tokens(container.get("required_capabilities") or container.get("capabilities"), lower=True))

    goal = event.get("goal")
    if isinstance(goal, str):
        for match in _CAPABILITY_TAG_RE.findall(goal):
            _extend(_normalise_module_tokens(_split_capability_tokens(match), lower=True))
    tasks = event.get("tasks")
    if isinstance(tasks, list):
        for task in tasks:
            if not isinstance(task, str):
                continue
            for match in _CAPABILITY_TAG_RE.findall(task):
                _extend(_normalise_module_tokens(_split_capability_tokens(match), lower=True))

    return required


class AgentState(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    IDLE = "idle"
    SLEEPING = "sleeping"
    TERMINATED = "terminated"


class AgentLifecycleManager:
    """Manage running agents and reload them when blueprints change."""

    def __init__(
        self,
        config: Config,
        llm_provider: ChatModelProvider,
        file_storage: FileStorage,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
        sleep_timeout: float = 3600.0,
    ) -> None:
        self._config = config
        self._llm_provider = llm_provider
        self._file_storage = file_storage
        self._event_bus = event_bus
        self._scheduler = scheduler
        if self._scheduler:
            self._scheduler.set_task_callback(self.notify_tasks)
        self._conductor = AgentConductor(event_bus)
        # Allow conductor to callback with metrics
        try:
            setattr(self._conductor, "_manager", self)
        except Exception:
            pass
        self._agents: Dict[str, Agent] = {}
        self._resources: Dict[str, Dict[str, float]] = {}
        self._states: Dict[str, AgentState] = {}
        self._heartbeats: Dict[str, float] = {}
        self._paths: Dict[str, Path] = {}
        self._task_count = 0
        self._heartbeat_timeout = 30.0
        self._sleep_timeout = sleep_timeout
        self._default_quota = {
            "cpu": float(os.getenv("AGENT_CPU_QUOTA", 80.0)),
            "memory": float(os.getenv("AGENT_MEMORY_QUOTA", 80.0)),
        }
        self._metrics = SystemMetricsCollector(event_bus)
        self._metrics.start()
        self._event_bus.subscribe("agent.resource", self._on_resource_event)
        self._event_bus.subscribe("agent.heartbeat", self._on_heartbeat)
        self._resource_stop = threading.Event()
        self._resource_thread = threading.Thread(
            target=self._resource_manager, daemon=True
        )
        self._resource_thread.start()
        self._watcher = BlueprintWatcher(self._on_blueprint_change)
        self._watcher.start()
        self._world_model = WorldModel()
        self._self_model = SelfModel()
        self._planner = Planner()
        self._goal_generator = GoalGenerator(
            world_model=self._world_model,
            event_bus=event_bus,
        )
        self._resource_scheduler = ResourceScheduler(
            global_workspace=global_workspace,
            event_bus=event_bus,
            scheduler=self._scheduler,
        )
        self._last_memory_hits = 0
        self._last_memory_check = time.time()
        self._retrain_requests: list[dict[str, Any]] = []
        listener = self._goal_generator.listener
        if listener:
            self._resource_scheduler.register_module(
                "goal_listener",
                listener.set_poll_interval,
                base_interval=listener.poll_interval,
                min_interval=5.0,
                max_interval=120.0,
                slowdown_factor=2.0,
                boost_factor=1.5,
            )
        self._backlog_lock = threading.Lock()
        self._scheduler_backlog = 0
        self._manager_backlog = 0
        self._planning_inflight = threading.Event()
        self._planning_handle: TaskHandle | None = None
        self._task_manager = TaskManager(
            event_bus=event_bus,
            queue_callback=self._on_task_queue_depth,
            resource_id=f"task-manager:{os.getpid()}",
        )
        cpu_workers = max(2, os.cpu_count() or 2)
        self._task_manager.configure_device("cpu", max_workers=cpu_workers)
        try:  # Optional GPU acceleration for specialised workloads
            import torch  # type: ignore

            if torch.cuda.is_available():
                gpu_workers = torch.cuda.device_count() or 1
                self._task_manager.configure_device("gpu", max_workers=gpu_workers)
        except Exception:
            pass
        self._task_manager.start()
        self._task_submitter = TaskSubmissionScheduler(
            task_manager=self._task_manager,
            event_bus=self._event_bus,
        )
        self._scheduler_control = None
        try:
            if SchedulerControlManager is not None:
                self._scheduler_control = SchedulerControlManager(
                    event_bus=self._event_bus,
                    task_manager=self._task_manager,
                    scheduler=self._scheduler,
                    logger_=logger,
                )
        except Exception:
            self._scheduler_control = None
        self._task_manager_env_adapter = None
        if str(os.getenv("TASK_MANAGER_DYNAMIC_CONCURRENCY", "")).strip().lower() in {"1", "true", "yes", "on"}:
            try:
                from modules.environment.environment_adapter import EnvironmentAdapter

                def _apply_env_adjustment(adjustment: Dict[str, Any]) -> None:
                    try:
                        concurrency = adjustment.get("concurrency")
                        if concurrency is None:
                            return
                        reason = adjustment.get("reason")
                        if self._scheduler_control is not None and bool(
                            getattr(self._scheduler_control, "enabled", True)
                        ):
                            try:
                                self._event_bus.publish(
                                    "scheduler.control",
                                    {
                                        "action": "throttle",
                                        "device": "cpu",
                                        "concurrency": int(concurrency),
                                        "reason": str(reason) if reason else None,
                                        "source": "environment_adapter",
                                    },
                                )
                                return
                            except Exception:
                                pass
                        self._task_manager.set_device_concurrency_limit(
                            "cpu",
                            int(concurrency),
                            reason=str(reason) if reason else None,
                            source="environment_adapter",
                        )
                    except Exception:
                        return

                adapter = EnvironmentAdapter.from_env(event_bus=event_bus, apply_callback=_apply_env_adjustment)
                if adapter is not None:
                    adapter.start()
                    self._task_manager_env_adapter = adapter
            except Exception:
                logger.debug("Failed to attach EnvironmentAdapter to TaskManager.", exc_info=True)
        self._knowledge_manager = KnowledgeConsolidator()
        self._memory_router = MemoryRouter(self._knowledge_manager)
        self._knowledge_importer = RuntimeKnowledgeImporter()
        self._knowledge_pipeline = KnowledgeUpdatePipeline(
            consolidator=self._knowledge_manager,
            importer=self._knowledge_importer,
            memory_router=self._memory_router,
        )
        self._knowledge_blindspot = None
        try:
            from .knowledge_blindspot import KnowledgeBlindspotDetector

            self._knowledge_blindspot = KnowledgeBlindspotDetector(memory_router=self._memory_router)
        except Exception:  # pragma: no cover - optional wiring
            self._knowledge_blindspot = None
        self._event_bus.subscribe("environment.perception", self._on_perception_event)
        self._event_bus.subscribe("agent.conductor.directive", self._on_decision_event)
        self._event_bus.subscribe("environment.demo", self._on_demonstration_event)
        self._event_bus.subscribe("agent.demonstration", self._on_demonstration_event)
        self._event_bus.subscribe("planner.plan_ready", self._on_plan_ready)
        self._event_bus.subscribe("upgrade.architecture.request", self._on_architecture_upgrade_request)
        self._event_bus.subscribe("automl.suggestion", self._on_automl_suggestion)
        self._event_bus.subscribe("agent.snapshot", self._on_manual_snapshot)
        module_adapters: list[ModuleAdapter] = []
        try:
            def _goal_listener_enabled() -> bool:
                listener = getattr(self._goal_generator, "listener", lambda: None)()
                thread = getattr(listener, "_thread", None)
                return bool(listener and thread and thread.is_alive())

            def _enable_goal_listener() -> None:
                listener = getattr(self._goal_generator, "listener", lambda: None)()
                if listener:
                    listener.start()
                    return
                from .goal_generator import GoalListener  # local import to avoid cycles

                self._goal_generator._listener = GoalListener(  # type: ignore[attr-defined]
                    self._world_model,
                    self._goal_generator,
                    event_bus=self._event_bus,
                    poll_interval=getattr(self._goal_generator, "listener_poll_interval", 30.0),
                    start=True,
                )

            def _disable_goal_listener() -> None:
                listener = getattr(self._goal_generator, "listener", lambda: None)()
                if listener:
                    try:
                        listener.stop()
                    except Exception:
                        pass
                    self._goal_generator._listener = None  # type: ignore[attr-defined]

            def _scale_goal_listener(interval_scale: float) -> None:
                listener = getattr(self._goal_generator, "listener", lambda: None)()
                if listener:
                    try:
                        base = getattr(self._goal_generator, "listener_poll_interval", 30.0)
                        listener.set_poll_interval(base * interval_scale)
                    except Exception:
                        pass
                    return
                try:
                    setattr(self._goal_generator, "listener_poll_interval", interval_scale * 30.0)
                except Exception:
                    pass

            module_adapters.append(
                ModuleAdapter(
                    name="goal_listener",
                    enable=_enable_goal_listener,
                    disable=_disable_goal_listener,
                    enabled_probe=_goal_listener_enabled,
                    scale_probe=lambda: getattr(self._goal_generator, "listener_poll_interval", 30.0) / 30.0,
                    apply_scale=_scale_goal_listener,
                    min_scale=0.2,
                    max_scale=3.0,
                )
            )
        except Exception:
            module_adapters = []
        try:
            arch_manager = HybridArchitectureManager.from_runtime(
                runtime_config=config,
                memory_manager=getattr(self._goal_generator, "memory", None),
                policy_module=getattr(self._planner, "policy", None),
                history=deque(maxlen=10),
                ga_config=GAConfig(population_size=8, generations=2, mutation_sigma=0.25),
                cooldown_steps=2,
                min_improvement=-0.01,
                seed=int(os.getenv("ARCH_EVOLUTION_SEED", "0")) if os.getenv("ARCH_EVOLUTION_SEED") else None,
                module_adapters=module_adapters,
                pso_bounds={
                    "module_goal_listener_scale": (0.2, 2.0),
                },
                pso_config={"num_particles": 6, "max_iter": 10},
            )
        except Exception:  # pragma: no cover - best-effort wiring
            arch_manager = None
        self._architecture_manager = arch_manager
        self._environment_loop = ActionPerceptionLoop(
            event_bus=event_bus,
            environment=GridWorldEnvironment(),
            knowledge_pipeline=self._knowledge_pipeline,
        )
        self._environment_loop.reset_environment()
        self._performance_monitor = None
        try:
            from backend.monitoring import MultiMetricMonitor

            self._performance_monitor = MultiMetricMonitor()
        except Exception:
            self._performance_monitor = None
        self._visualizer = None
        try:
            from BrainSimulationSystem.visualization.visualizer import BrainVisualizer  # type: ignore

            self._visualizer = BrainVisualizer(self._world_model)
        except Exception:
            self._visualizer = None
        self._visualizer_metrics: Dict[str, float] = {}
        self._learning_model_lock = threading.RLock()
        self._predictive_model = None
        backend = str(os.getenv("PREDICTIVE_MODEL_BACKEND", "numpy") or "numpy").strip().lower()

        def _env_bool(name: str, default: bool = False) -> bool:
            value = os.getenv(name)
            if value is None:
                return bool(default)
            return str(value).strip().lower() in {"1", "true", "yes", "on"}

        def _apply_predictor_overrides(cfg: Any) -> None:
            if cfg is None:
                return
            if hasattr(cfg, "lr_scheduler_enabled"):
                try:
                    cfg.lr_scheduler_enabled = _env_bool(
                        "SELF_SUPERVISED_LR_SCHEDULER",
                        getattr(cfg, "lr_scheduler_enabled", False),
                    )
                except Exception:
                    pass
            for env_name, attr in (
                ("SELF_SUPERVISED_LR_TARGET_LOSS", "lr_target_loss"),
                ("SELF_SUPERVISED_LR_DECAY", "lr_decay"),
                ("SELF_SUPERVISED_LR_GROWTH", "lr_growth"),
                ("SELF_SUPERVISED_LR_MIN", "lr_min"),
                ("SELF_SUPERVISED_LR_MAX", "lr_max"),
            ):
                value = os.getenv(env_name)
                if value is None or not hasattr(cfg, attr):
                    continue
                try:
                    setattr(cfg, attr, float(value))
                except Exception:
                    continue
        if backend == "torch":
            try:
                from BrainSimulationSystem.learning.torch_self_supervised import (  # type: ignore
                    TorchSelfSupervisedConfig,
                    TorchSelfSupervisedPredictor,
                )
            except Exception:
                TorchSelfSupervisedPredictor = None  # type: ignore[assignment]
                TorchSelfSupervisedConfig = None  # type: ignore[assignment]
            if TorchSelfSupervisedPredictor is not None:
                try:
                    cfg = TorchSelfSupervisedConfig() if TorchSelfSupervisedConfig is not None else None
                    _apply_predictor_overrides(cfg)
                    self._predictive_model = TorchSelfSupervisedPredictor(cfg)
                except Exception:
                    self._predictive_model = None
        if self._predictive_model is None and SelfSupervisedPredictor is not None:
            try:
                cfg = SelfSupervisedConfig() if SelfSupervisedConfig is not None else None  # type: ignore[call-arg]
                _apply_predictor_overrides(cfg)
                self._predictive_model = SelfSupervisedPredictor(cfg)  # type: ignore[call-arg]
            except Exception:
                self._predictive_model = None
        self._last_action_metadata: Dict[str, Any] = {}
        self._imitation_buffer: deque[dict] = deque(maxlen=256)
        self._imitation_policy = None
        try:  # Optional: behaviour cloning policy trained from demonstrations
            from modules.learning.behavior_cloning import BehaviorCloningConfig, BehaviorCloningPolicy

            enabled = str(os.getenv("IMITATION_BC_ENABLED", "1")).strip().lower() in {"1", "true", "yes", "on"}
            if enabled:
                cfg = BehaviorCloningConfig(
                    state_dim=int(os.getenv("IMITATION_STATE_DIM", "64")),
                    lr=float(os.getenv("IMITATION_LR", "0.05")),
                    weight_decay=float(os.getenv("IMITATION_WEIGHT_DECAY", "0.0001")),
                    label_smoothing=float(os.getenv("IMITATION_LABEL_SMOOTHING", "0.05")),
                    entropy_bonus=float(os.getenv("IMITATION_ENTROPY_BONUS", "0.01")),
                    action_vocab_limit=int(os.getenv("IMITATION_ACTION_VOCAB_LIMIT", "128")),
                    inference_temperature=float(os.getenv("IMITATION_INFERENCE_TEMPERATURE", "1.0")),
                    inference_uniform_mix=float(os.getenv("IMITATION_UNIFORM_MIX", "0.05")),
                    seed=int(os.getenv("IMITATION_SEED", "0")) if os.getenv("IMITATION_SEED") else None,
                )
                self._imitation_policy = BehaviorCloningPolicy(cfg)
        except Exception:  # pragma: no cover - keep runtime optional
            self._imitation_policy = None
        self._last_perception_event: Dict[str, Any] = {}
        self._working_memory: deque[dict] = deque(maxlen=64)
        self._working_memory_ttl = 300.0  # seconds
        self._learning_manager = LearningManager(
            event_bus=event_bus,
            task_manager=self._task_manager,
            run_learning_cycle=self._run_learning_cycle,
            logger_=logger,
        )
        self._learning_guard = LearningSafetyGuard(logger_=logger)
        self._self_debug_manager = None
        try:
            self._self_debug_manager = SelfDebugManager(
                event_bus=event_bus,
                memory_router=self._memory_router,
                performance_monitor=self._performance_monitor,
                logger_=logger,
            )
        except Exception:
            self._self_debug_manager = None
        self._self_correction_manager = None
        try:
            self._self_correction_manager = SelfCorrectionManager(
                event_bus=event_bus,
                memory_router=self._memory_router,
                performance_monitor=self._performance_monitor,
                logger_=logger,
            )
        except Exception:
            self._self_correction_manager = None
        self._self_diagnoser = None
        try:
            self._self_diagnoser = SelfDiagnoser.from_env(
                event_bus=event_bus,
                memory_router=self._memory_router,
                performance_monitor=self._performance_monitor,
                logger_=logger,
            )
        except Exception:
            self._self_diagnoser = None
        self._self_reflection = None
        try:
            from .self_reflection import SelfReflectionLoop

            self._self_reflection = SelfReflectionLoop.from_env(
                event_bus=event_bus,
                memory_router=self._memory_router,
                logger_=logger,
            )
        except Exception:
            self._self_reflection = None
        self._automl_manager = None
        try:
            self._automl_manager = AutoMLManager(event_bus=event_bus, task_manager=self._task_manager, logger_=logger)
        except Exception:
            self._automl_manager = None
        self._code_self_repair = None
        try:
            self._code_self_repair = CodeSelfRepairManager(event_bus=event_bus, task_manager=self._task_manager, logger_=logger)
        except Exception:
            self._code_self_repair = None
        self._capability_gap_threshold = float(os.getenv("CAPABILITY_GAP_THRESHOLD", "0.5"))
        self._last_checkpoint_loaded = False
        self._self_improvement = SelfImprovementManager() if SelfImprovementManager is not None else None
        self._confidence_threshold = float(os.getenv("INTERNAL_REVIEW_THRESHOLD", "0.5"))
        self._confidence_weights = (
            float(os.getenv("CONF_WEIGHT_PERCEPTION", "0.6")),
            float(os.getenv("CONF_WEIGHT_DECISION", "0.4")),
        )
        self._last_perception_confidence: float | None = None
        self._last_decision_confidence: float | None = None
        self._last_review_ts: float = 0.0
        self._cross_test_interval = float(os.getenv("CROSS_DOMAIN_TEST_INTERVAL", "1800"))
        self._last_cross_test_ts = 0.0
        self._cross_domain_benchmark = CrossDomainBenchmark() if CrossDomainBenchmark is not None else None
        self._cross_domain_every_n_cycles = int(os.getenv("CROSS_DOMAIN_EVERY_N_CYCLES", "3"))
        self._learning_cycle_count = 0
        self._health_interval = float(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
        self._health_mem_threshold = float(os.getenv("HEALTH_MEM_THRESHOLD", "80"))
        self._health_cpu_threshold = float(os.getenv("HEALTH_CPU_THRESHOLD", "95"))
        self._health_stop = threading.Event()
        self._health_thread = threading.Thread(
            target=self._health_monitor, name="agent-health", daemon=True
        )
        self._health_thread.start()
        self._internal_tasks: deque[dict] = deque()
        self._internal_task_interval = float(os.getenv("INTERNAL_TASK_INTERVAL", "3600"))
        self._last_internal_task_ts = 0.0
        self._checkpoint_interval = float(os.getenv("CHECKPOINT_INTERVAL", "3600"))
        self._last_checkpoint_ts = 0.0
        self._last_checkpoint_loaded = False
        self._load_latest_checkpoint()

        self._adaptive_controller = AdaptiveResourceController(
            config=config,
            event_bus=event_bus,
            memory_router=self._memory_router,
            long_term_memory=getattr(self._goal_generator, "memory", None),
            logger=logger,
            architecture_manager=arch_manager,
            monitor=self._performance_monitor,
            meta_adjustment_provider=self._meta_adjust_parameters,
            retrain_callback=self._schedule_retrain,
            resource_optimizer=self._optimize_resources,
            self_improvement_manager=self._self_improvement,
        )
        self._event_bus.subscribe(
            "coordinator.task_completed", self._on_task_completed
        )
        self._event_bus.subscribe("agent.action.outcome", self._on_action_outcome)
        # Manage capability modules requested at runtime by agents
        self._module_manager = RuntimeModuleManager(event_bus)
        control = getattr(self, "_scheduler_control", None)
        attach = getattr(control, "attach_module_manager", None) if control is not None else None
        if callable(attach):
            try:
                attach(self._module_manager)
            except Exception:
                pass
        self._module_acquisition = (
            ModuleAcquisitionManager(
                module_manager=self._module_manager,
                task_manager=self._task_manager,
                event_bus=self._event_bus,
            )
            if ModuleAcquisitionManager is not None
            else None
        )
        self._upgrade_decision = None
        try:
            if UpgradeDecisionEngine is not None:
                self._upgrade_decision = UpgradeDecisionEngine(
                    event_bus=self._event_bus,
                    logger_=logger,
                )
        except Exception:
            self._upgrade_decision = None
        self._module_lifecycle = None
        try:
            if ModuleLifecycleManager is not None:
                self._module_lifecycle = ModuleLifecycleManager(
                    event_bus=self._event_bus,
                    module_manager=self._module_manager,
                    logger_=logger,
                )
        except Exception:
            self._module_lifecycle = None
        self._fault_recovery = None
        try:
            if FaultRecoveryManager is not None:
                self._fault_recovery = FaultRecoveryManager(
                    event_bus=self._event_bus,
                    module_manager=self._module_manager,
                    logger_=logger,
                )
        except Exception:
            self._fault_recovery = None
        plugin_paths = os.getenv("SKILL_PLUGIN_PATHS")
        if plugin_paths:
            paths = [
                Path(p.strip())
                for p in plugin_paths.split(os.pathsep)
                if p.strip()
            ]
            if paths:
                for plugin_path in paths:
                    try:
                        resolved = plugin_path.resolve()
                    except Exception:
                        resolved = plugin_path
                    if resolved.exists() and str(resolved) not in sys.path:
                        sys.path.insert(0, str(resolved))
                refresh_skills_from_directory(paths, prune_missing=False)
                try:
                    registry = get_skill_registry()
                    logger.info(
                        "Skill registry initialised with %d skills from %s",
                        len(registry.list_specs()),
                        plugin_paths,
                    )
                except Exception:
                    logger.debug("Skill registry initialisation logging failed.", exc_info=True)
                try:
                    from modules.environment import get_hardware_registry  # late import

                    hardware_snapshot = get_hardware_registry().snapshot()
                    if hardware_snapshot:
                        logger.info(
                            "Initial hardware registry snapshot: %s",
                            {
                                worker: caps.get("hardware", {}).get("name")
                                for worker, caps in hardware_snapshot.items()
                                if isinstance(caps, dict)
                            },
                        )
                except Exception:
                    logger.debug("Hardware registry snapshot unavailable during bootstrap.", exc_info=True)

        # Shared async runtime for I/O bound helpers ---------------------------------
        self._async_executor = ThreadPoolExecutor(
            max_workers=max(4, cpu_workers * 2),
            thread_name_prefix="agent-io",
        )
        self._async_loop = asyncio.new_event_loop()
        self._async_loop.set_exception_handler(self._handle_async_exception)

        def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self._async_loop_thread = threading.Thread(
            target=_run_loop,
            args=(self._async_loop,),
            daemon=True,
            name="agent-async-loop",
        )
        self._async_loop_thread.start()

        global_workspace.register_module("async.executor", self._async_executor)
        global_workspace.broadcast("async.executor", self._async_executor)
        global_workspace.register_module("async.loop", self._async_loop)
        global_workspace.broadcast("async.loop", self._async_loop)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _set_state(self, name: str, state: AgentState) -> None:
        if self._states.get(name) == state:
            return
        self._states[name] = state
        self._event_bus.publish(
            "agent.state", {"agent": name, "state": state.value, "time": time.time()}
        )

    async def _on_heartbeat(self, event: Dict[str, Any]) -> None:
        name = event.get("agent")
        if not name:
            return
        self._heartbeats[name] = event.get("time", time.time())
        if self._states.get(name) in {AgentState.SLEEPING, AgentState.IDLE}:
            self._set_state(name, AgentState.RUNNING)

    async def _on_task_completed(self, event: Dict[str, Any]) -> None:
        """Capture completed task details for downstream knowledge consolidation."""
        detail = event.get("detail")
        if isinstance(detail, str) and detail.strip():
            source = event.get("task_id") or "task"
            metadata = {"agent": event.get("agent_id"), "status": event.get("status")}
            self._memory_router.add_observation(
                detail, source=f"task:{source}", metadata=metadata
            )
        summary = event.get("summary")
        if isinstance(summary, str) and summary.strip():
            source = event.get("task_id") or "task"
            metadata = {"agent": event.get("agent_id"), "summary": True}
            self._memory_router.add_observation(
                summary, source=f"task:{source}", metadata=metadata
            )
        if hasattr(self, "_knowledge_pipeline"):
            try:
                self._knowledge_pipeline.process_task_event(event)
            except Exception:  # pragma: no cover - defensive logging
                logger.debug("Knowledge update pipeline failed.", exc_info=True)
        reflector = getattr(self, "_self_reflection", None)
        if reflector is not None:
            try:
                trace = list(getattr(self, "_working_memory", []) or [])[-10:]
                executor = getattr(self, "_async_executor", None)
                if executor is not None:
                    executor.submit(reflector.reflect_task, dict(event), trace=trace)
                else:
                    reflector.reflect_task(dict(event), trace=trace)
            except Exception:
                logger.debug("Self-reflection loop failed.", exc_info=True)

    async def _on_decision_event(self, event: Dict[str, Any]) -> None:
        """Update visualizer/monitor with decision directive flows."""

        if not isinstance(event, dict):
            return
        try:
            self._performance_monitor.log_snapshot({"decision_replans": 1.0})
        except Exception:
            pass
        self._update_visualizer_metrics({"decision_replans": 1.0})

    async def _on_manual_snapshot(self, event: Dict[str, Any]) -> None:
        """Handle manual snapshot requests via event bus."""

        try:
            reason = event.get("reason", "manual")
            self._checkpoint_state(reason=reason)
        except Exception:
            logger.debug("Manual snapshot failed.", exc_info=True)

    async def _on_demonstration_event(self, event: Dict[str, Any]) -> None:
        """Capture demonstration samples for imitation learning."""

        if not isinstance(event, dict):
            return
        action = event.get("action") or event.get("command")
        state = event.get("state") or dict(self._last_perception_event)
        if not action or not isinstance(state, dict):
            return
        domains = list(self._extract_domains([state, event.get("metadata"), action]))
        sample = {
            "state": state,
            "action": action,
            "reward": event.get("reward"),
            "trained": False,
            "metadata": {
                **{k: v for k, v in event.items() if k not in {"state", "action", "command"}},
                "domains": domains,
            },
            "timestamp": time.time(),
        }
        self._imitation_buffer.append(sample)
        try:
            self._event_bus.publish("learning.request", {"reason": "demonstration"})
        except Exception:
            pass
        snapshot = {"imitation_buffer": float(len(self._imitation_buffer))}
        try:
            self._adaptive_controller.record_extra_metrics(snapshot)
        except Exception:
            pass
        if self._performance_monitor is not None:
            try:
                self._performance_monitor.log_snapshot(snapshot)
            except Exception:
                pass
        self._update_visualizer_metrics(snapshot)
        self._record_working_memory({"type": "imitation", "sample": sample, "timestamp": sample["timestamp"]})

    async def _on_plan_ready(self, event: Dict[str, Any]) -> None:
        """Assess knowledge coverage for new plans/goals."""

        if not isinstance(event, dict):
            return
        goal = event.get("goal") or ""
        tasks = event.get("tasks") or []
        required_modules = _extract_required_modules_from_plan_event(event)
        if required_modules:
            prune = os.getenv("MODULE_MANAGER_PRUNE_ON_PLAN_READY", "").strip().lower() in {"1", "true", "yes", "on"}
            try:
                if prune:
                    self._module_manager.update(required_modules, prune=True)
                else:
                    self._module_manager.ensure(required_modules)
            except Exception:
                logger.debug("Failed to update capability modules for plan_ready.", exc_info=True)
        domains = self._extract_domains([goal] + tasks if isinstance(tasks, list) else [goal])
        gaps = self._assess_knowledge_gaps(domains)

        query_parts = [str(goal or "").strip()]
        if isinstance(tasks, list):
            query_parts.extend(str(t or "").strip() for t in tasks[:8])
        query_text = "\n".join(part for part in query_parts if part)

        blindspot = None
        source = str(event.get("source") or "").strip().lower()
        if self._knowledge_blindspot is not None and source not in {"self_improvement", "cross_domain_test"}:
            try:
                blindspot = self._knowledge_blindspot.assess(query_text=query_text, keywords=sorted(domains))
            except Exception:
                blindspot = None

        if blindspot is not None and getattr(blindspot, "blindspot", False):
            gaps = set(gaps) | set(domains)
            try:
                payload = {
                    "time": time.time(),
                    "goal": goal,
                    "tasks": tasks if isinstance(tasks, list) else [],
                    "domains": sorted(domains),
                    "assessment": blindspot.to_dict() if hasattr(blindspot, "to_dict") else dict(blindspot),
                    "declaration": {
                        "zh": getattr(blindspot, "declaration_zh", "") or "我目前缺乏这方面知识",
                        "en": getattr(blindspot, "declaration_en", "") or "I currently lack sufficient knowledge about this topic.",
                    },
                    "source": "knowledge_blindspot",
                }
                self._event_bus.publish("diagnostics.knowledge_blindspot", payload)
            except Exception:
                pass
            try:
                self._record_working_memory(
                    {
                        "type": "knowledge_blindspot",
                        "goal": goal,
                        "tasks": tasks if isinstance(tasks, list) else [],
                        "domains": sorted(domains),
                        "warning": getattr(blindspot, "declaration_zh", "") or "我目前缺乏这方面知识",
                        "timestamp": time.time(),
                    }
                )
            except Exception:
                pass
            try:
                self._event_bus.publish(
                    "learning.request",
                    {"reason": "knowledge_blindspot", "domains": sorted(domains), "time": time.time()},
                )
            except Exception:
                pass
        reflection_hints: list[dict[str, Any]] = []
        reflector = getattr(self, "_self_reflection", None)
        if reflector is not None and query_text:
            try:
                reflection_hints = reflector.retrieve_hints(query_text)
            except Exception:
                reflection_hints = []
        if reflection_hints:
            now = time.time()
            payload = {
                "time": now,
                "goal": goal,
                "tasks": tasks if isinstance(tasks, list) else [],
                "domains": sorted(domains),
                "query": query_text,
                "hints": reflection_hints,
            }
            try:
                self._event_bus.publish("diagnostics.self_reflection_hints", payload)
            except Exception:
                pass
            try:
                self._record_working_memory(
                    {
                        "type": "self_reflection_hints",
                        "goal": goal,
                        "domains": sorted(domains),
                        "hints": reflection_hints,
                        "timestamp": now,
                    }
                )
            except Exception:
                pass
        if not gaps:
            return
        summary = f"knowledge gaps: {', '.join(sorted(gaps))}"
        try:
            self._memory_router.add_observation(summary, source="self_model", metadata={"gaps": list(gaps)})
        except Exception:
            logger.debug("Failed to persist knowledge gap summary", exc_info=True)
        payload = {"knowledge_gaps": float(len(gaps))}
        try:
            self._adaptive_controller.record_extra_metrics(payload)
        except Exception:
            pass
        if self._performance_monitor is not None:
            try:
                self._performance_monitor.log_snapshot(payload)
            except Exception:
                pass
        self._update_visualizer_metrics(payload)
        self._update_self_improvement_from_metrics({**payload, "knowledge_gap_domains": list(gaps), "goal": goal, "tasks": tasks})
        self._record_working_memory(
            {"type": "knowledge_gap", "gaps": list(gaps), "goal": goal, "timestamp": time.time()}
        )
        if self._module_acquisition is not None and required_modules:
            try:
                self._module_acquisition.request_for_tasks(
                    required_modules,
                    goal=str(goal or ""),
                    reason="plan_ready_requirements",
                    context={"knowledge_gap_domains": list(gaps), "source": source},
                )
            except Exception:
                logger.debug("Module acquisition request failed for plan_ready.", exc_info=True)
            if os.getenv("MODULE_ACQUISITION_RESEARCH_KNOWLEDGE_GAPS", "").strip().lower() in {"1", "true", "yes", "on"}:
                for domain in sorted(gaps):
                    try:
                        self._module_acquisition.request(
                            f"python library for {domain}",
                            context={"goal": goal, "domain": domain, "reason": "knowledge_gap"},
                        )
                    except Exception:
                        logger.debug("Module acquisition request failed for domain=%s", domain, exc_info=True)

    async def _on_architecture_upgrade_request(self, event: Dict[str, Any]) -> None:
        """Request a (possibly gated) architecture evolution step."""

        if not isinstance(event, dict):
            return
        controller = getattr(self, "_adaptive_controller", None)
        if controller is None:
            return
        try:
            steps = int(event.get("steps", 1) or 1)
        except Exception:
            steps = 1
        steps = max(1, steps)
        reason = str(event.get("reason") or event.get("source") or "upgrade_decision").strip() or "upgrade_decision"
        try:
            requested = controller.request_architecture_evolution(reason=reason, steps=steps)
        except Exception:
            requested = False
        if not requested:
            return
        try:
            self._event_bus.publish(
                "upgrade.architecture.accepted",
                {"time": time.time(), "reason": reason, "steps": steps},
            )
        except Exception:
            pass

    async def _on_automl_suggestion(self, event: Dict[str, Any]) -> None:
        """Accept AutoML hyper-parameter suggestions for closed-loop evaluation."""

        if self._self_improvement is None or not isinstance(event, dict):
            return
        try:
            queued = self._self_improvement.enqueue_automl_suggestion(event)
        except Exception:
            logger.debug("Failed to enqueue AutoML suggestion", exc_info=True)
            return
        if not queued:
            return
        try:
            self._event_bus.publish("learning.request", {"reason": "automl_suggestion"})
        except Exception:
            pass
        if self._performance_monitor is not None:
            try:
                self._performance_monitor.log_snapshot({"automl_suggestion": 1.0})
            except Exception:
                pass
        self._record_working_memory({"type": "automl_suggestion", "event": dict(event), "timestamp": time.time()})

    async def _on_action_outcome(self, event: Dict[str, Any]) -> None:
        """Capture executed actions for predictive world-model training."""

        if not isinstance(event, dict):
            return
        if self._predictive_model is None:
            return
        action = event.get("command") or event.get("action")
        reward = event.get("reward")
        metadata = {
            "agent": event.get("agent"),
            "reward": reward,
            "status": event.get("status"),
            "confidence": event.get("confidence"),
        }
        try:
            self._predictive_model.record_action(action, metadata=metadata)  # type: ignore[call-arg]
            self._last_action_metadata = {"action": action, "reward": reward, "metadata": metadata}
        except Exception:
            self._last_action_metadata = {}
            logger.debug("Predictive model failed to record action.", exc_info=True)
        # Persist episodic experience tying perception, action, and reward.
        self._store_episodic_experience(
            perception=self._last_perception_event,
            action=action,
            reward=reward,
            status=event.get("status"),
        )
        status = str(event.get("status") or "").lower()
        reward_value = None
        try:
            reward_value = float(reward) if reward is not None else None
        except Exception:
            reward_value = None
        if status == "error" or (reward_value is not None and reward_value < 0):
            self._reflect_failure(event, action=action, reward=reward_value)

    async def _on_perception_event(self, event: Dict[str, Any]) -> None:
        """Log perception confidence and prediction signals for monitoring."""

        if isinstance(event, dict):
            self._last_perception_event = dict(event)
            self._record_working_memory(
                {
                    "type": "perception",
                    "data": event,
                    "timestamp": time.time(),
                }
            )
            self._last_perception_ts = time.time()
        annotations = event.get("annotations") or event.get("semantic_annotations") or {}
        confidences: list[float] = []
        prediction_errors: list[float] = []
        for data in annotations.values():
            if not isinstance(data, dict):
                continue
            conf = data.get("confidence")
            if conf is not None:
                try:
                    confidences.append(float(conf))
                except Exception:
                    pass
            ss = data.get("self_supervised")
            if isinstance(ss, dict):
                err = ss.get("prediction_error")
                if err is not None:
                    try:
                        prediction_errors.append(float(err))
                    except Exception:
                        pass
        snapshot: Dict[str, float] = {}
        if confidences:
            snapshot["perception_confidence_avg"] = sum(confidences) / max(len(confidences), 1)
            self._last_perception_confidence = snapshot["perception_confidence_avg"]
        if prediction_errors:
            snapshot["perception_prediction_error"] = sum(prediction_errors) / max(len(prediction_errors), 1)
        if snapshot:
            try:
                self._adaptive_controller.record_module_metric(
                    "perception",
                    throughput=len(annotations) or None,
                    latency=None,
                )
            except Exception:
                pass
            try:
                self._adaptive_controller.record_extra_metrics(snapshot)
            except Exception:
                pass
            if self._performance_monitor is not None:
                try:
                    self._performance_monitor.log_snapshot(snapshot)
                except Exception:
                    pass
            self._update_visualizer_metrics(snapshot)
            self._update_self_improvement_from_metrics(snapshot)

        # Train/update predictive world model on raw perception stream.
        if self._predictive_model is not None:
            try:
                metadata = {"timestamp": time.time()}
                if self._last_action_metadata:
                    metadata.update(self._last_action_metadata.get("metadata", {}))
                with self._learning_model_lock:
                    ss_summary = self._predictive_model.observe(event or {}, metadata=metadata)  # type: ignore[call-arg]
                extras: Dict[str, float] = {}
                pred_err = ss_summary.get("prediction_error")
                if pred_err is not None:
                    extras["perception_prediction_error"] = float(pred_err)
                recon = ss_summary.get("reconstruction_loss")
                if recon is not None:
                    extras["perception_reconstruction_loss"] = float(recon)
                if extras:
                    try:
                        self._adaptive_controller.record_extra_metrics(extras)
                    except Exception:
                        pass
                    if self._performance_monitor is not None:
                        try:
                            self._performance_monitor.log_snapshot(extras)
                        except Exception:
                            pass
                    self._update_visualizer_metrics(extras)
                    self._update_self_improvement_from_metrics(extras)
                self._last_action_metadata = {}
            except Exception:
                logger.debug("Predictive model observation failed.", exc_info=True)

    def notify_tasks(self, count: int) -> None:
        """Update the current number of pending tasks."""
        self._task_count = count
        with self._backlog_lock:
            self._scheduler_backlog = max(0, count)
        self._update_backlog_pressure()

    def _on_task_queue_depth(self, depth: int) -> None:
        with self._backlog_lock:
            self._manager_backlog = max(0, depth)
        self._update_backlog_pressure()

    def _update_backlog_pressure(self) -> None:
        if not self._resource_scheduler:
            return
        with self._backlog_lock:
            total = self._scheduler_backlog + self._manager_backlog
        self._resource_scheduler.update_backlog(total)
        self._record_memory_hits()

    def _record_memory_hits(self) -> None:
        """Log memory hit rate to performance monitor and controller."""

        now = time.time()
        hits = 0
        try:
            hits = int(get_memory_hits())
        except Exception:
            hits = 0
        delta_hits = max(0, hits - self._last_memory_hits)
        dt = max(1e-3, now - self._last_memory_check)
        rate = delta_hits / dt
        self._last_memory_hits = hits
        self._last_memory_check = now
        if delta_hits and self._performance_monitor is not None:
            try:
                self._performance_monitor.log_snapshot({"memory_hit_rate": rate, "memory_hits": hits})
            except Exception:
                pass
        self._adaptive_controller.record_extra_metrics({"memory_hit_rate": rate})
        if self._visualizer is not None:
            try:
                snapshot = {
                    "memory_hit_rate": rate,
                    "memory_hits": hits,
                    "avg_cpu": float(self._resources.get("manager", {}).get("cpu", 0.0)),
                    "avg_memory": float(self._resources.get("manager", {}).get("memory", 0.0)),
                }
                if hasattr(self._visualizer, "update_metrics"):
                    self._visualizer.update_metrics(snapshot)  # type: ignore[attr-defined]
            except Exception:
                pass
        try:
            self._adaptive_controller.record_module_metric("memory", throughput=rate)
            # Feed decision success metrics when available.
            for engine in self._conductor._decision_engines.values():  # type: ignore[attr-defined]
                metrics = getattr(engine, "metrics", {})
                success_rate = metrics.get("success_rate")
                avg_reward = metrics.get("reward_avg")
                if success_rate is not None:
                    self._adaptive_controller.record_module_metric(
                        "decision",
                        throughput=metrics.get("decisions"),
                        latency=None,
                    )
                    if self._performance_monitor is not None:
                        self._performance_monitor.log_snapshot(
                            {
                                "decision_success_rate": float(success_rate),
                                **(
                                    {"decision_reward_avg": float(avg_reward)}
                                    if avg_reward is not None
                                    else {}
                                ),
                            }
                        )
                    self._update_visualizer_metrics(
                        {
                            "decision_success_rate": float(success_rate),
                            **(
                                {"decision_reward_avg": float(avg_reward)}
                                if avg_reward is not None
                                else {}
                            ),
                        }
                    )
                    self._adaptive_controller.record_extra_metrics(
                        {
                            "decision_success_rate": float(success_rate),
                            **(
                                {"decision_reward_avg": float(avg_reward)}
                                if avg_reward is not None
                                else {}
                            ),
                        }
                    )
                    self._last_decision_confidence = float(success_rate)
        except Exception:
            pass

    def _update_visualizer_metrics(self, payload: Dict[str, float]) -> None:
        """Store and forward metrics to the visualizer if present."""

        if not payload:
            return
        self._visualizer_metrics.update(payload)
        if self._visualizer is not None and hasattr(self._visualizer, "update_metrics"):
            try:
                self._visualizer.update_metrics(self._visualizer_metrics)  # type: ignore[attr-defined]
            except Exception:
                pass

    def _health_monitor(self) -> None:
        """Background health watchdog for long-running stability."""

        while not self._health_stop.wait(self._health_interval):
            try:
                mem_pct = None
                cpu_pct = None
                if psutil is not None:
                    process = psutil.Process(os.getpid())
                    mem_pct = process.memory_percent()
                    cpu_pct = psutil.cpu_percent(interval=0.05)
                if mem_pct is not None and mem_pct >= self._health_mem_threshold:
                    gc.collect()
                    try:
                        self._memory_router.shrink(max_age=900.0, min_usage=1)
                    except Exception:
                        logger.debug("Health shrink failed", exc_info=True)
                    # Unload optional predictive model if idle to free memory.
                    idle_secs = time.time() - getattr(self, "_last_perception_ts", time.time())
                    unload_after = float(os.getenv("MODEL_UNLOAD_IDLE_SECS", "1800"))
                    if idle_secs >= unload_after:
                        self._predictive_model = None
                if cpu_pct is not None and cpu_pct >= self._health_cpu_threshold:
                    # Slow down background learning by consuming tokens.
                    if getattr(self, "_learning_manager", None) is not None:
                        try:
                            self._learning_manager.throttle(max_tokens=0.5)
                        except Exception:
                            pass
                    try:
                        self._event_bus.publish("agent.resource", {"action": "throttle", "reason": "high_cpu"})
                    except Exception:
                        pass
                if self._performance_monitor is not None:
                    snapshot = {}
                    if mem_pct is not None:
                        snapshot["health_mem_pct"] = float(mem_pct)
                    if cpu_pct is not None:
                        snapshot["health_cpu_pct"] = float(cpu_pct)
                    if snapshot:
                        self._performance_monitor.log_snapshot(snapshot)
                if self._module_lifecycle is not None:
                    try:
                        self._module_lifecycle.evaluate()
                    except Exception:
                        logger.debug("Module lifecycle evaluation failed", exc_info=True)
            except Exception:
                logger.debug("Health monitor tick failed", exc_info=True)

    def _update_self_improvement_from_metrics(self, metrics: Mapping[str, Any]) -> None:
        """Feed decision metrics into the self-improvement goal manager."""

        if self._self_improvement is None:
            return
        try:
            if "decision_success_rate" in metrics:
                self._self_improvement.ensure_goal("decision_success_rate", 0.8, direction="increase")
            if "decision_reward_avg" in metrics:
                self._self_improvement.ensure_goal("decision_reward_avg", 0.0, direction="increase")
            if "perception_confidence_avg" in metrics:
                target = float(os.getenv("PERCEPTION_CONFIDENCE_TARGET", "0.6"))
                self._self_improvement.ensure_goal("perception_confidence_avg", target, direction="increase")
            if "perception_prediction_error" in metrics:
                target = float(os.getenv("PERCEPTION_PREDICTION_ERROR_TARGET", "0.25"))
                self._self_improvement.ensure_goal("perception_prediction_error", target, direction="decrease")
            if "knowledge_gaps" in metrics:
                try:
                    patience = int(float(os.getenv("SELF_IMPROVEMENT_KNOWLEDGE_PATIENCE", "1") or 1))
                except Exception:
                    patience = 1
                try:
                    cooldown = float(os.getenv("SELF_IMPROVEMENT_KNOWLEDGE_COOLDOWN_SECS", "300") or 300.0)
                except Exception:
                    cooldown = 300.0
                self._self_improvement.ensure_goal(
                    "knowledge_gaps",
                    0.0,
                    direction="decrease",
                    patience=patience,
                    cooldown_secs=cooldown,
                )
            self._self_improvement.observe_metrics(metrics)
        except Exception:
            logger.debug("Self-improvement metric update failed", exc_info=True)

    def _evaluate_global_confidence(self) -> None:
        """Aggregate confidence signals and trigger internal review when low."""

        now = time.time()
        perception_conf = self._last_perception_confidence
        decision_conf = self._last_decision_confidence
        if perception_conf is None and decision_conf is None:
            return
        w_p, w_d = self._confidence_weights
        total_weight = (w_p if perception_conf is not None else 0.0) + (w_d if decision_conf is not None else 0.0)
        if total_weight <= 0:
            return
        combined = 0.0
        if perception_conf is not None:
            combined += w_p * perception_conf
        if decision_conf is not None:
            combined += w_d * decision_conf
        combined /= total_weight
        if combined >= self._confidence_threshold:
            return
        if now - self._last_review_ts < 30.0:
            return
        self._last_review_ts = now
        payload = {
            "confidence": combined,
            "perception_confidence_avg": perception_conf,
            "decision_confidence": decision_conf,
            "threshold": self._confidence_threshold,
        }
        try:
            self._event_bus.publish("metacognition.review", payload)
        except Exception:
            logger.debug("Failed to publish metacognition review request", exc_info=True)
        if self._adaptive_controller is not None:
            try:
                self._adaptive_controller.record_extra_metrics({"global_confidence": combined})
            except Exception:
                pass
        if self._performance_monitor is not None:
            try:
                self._performance_monitor.log_snapshot({"global_confidence": combined})
            except Exception:
                pass
        self._update_visualizer_metrics({"global_confidence": combined})

    def _maybe_generate_internal_goal(self, backlog: int) -> None:
        """Spawn self-improvement goals when idle to drive autonomous practice."""

        if self._self_improvement is None:
            return
        if backlog > 0:
            return
        goal = self._self_improvement.generate_goal()
        if not goal:
            return
        goal_text = f"Improve {goal['name']} towards {goal['target']:.3f} ({goal['direction']})"
        self._event_bus.publish("planner.plan_ready", {"goal": goal_text, "tasks": [], "source": "self_improvement"})

    def _run_learning_cycle(self) -> Dict[str, Any]:
        """Background learning dispatcher: imitation replay and self-supervised refresh."""

        stats: Dict[str, Any] = {
            "imitation_pending": len(self._imitation_buffer),
            "working_memory": len(self._working_memory),
        }
        stats.update(self._consolidate_memories())
        stats.update(self._maybe_queue_cross_domain_self_test())
        if self._learning_cycle_count % max(1, self._cross_domain_every_n_cycles) == 0:
            stats.update(self._run_cross_domain_evaluator())
        self._learning_cycle_count += 1
        with self._learning_model_lock:
            stats.update(
                apply_online_model_updates(
                    self._predictive_model,
                    self._working_memory,
                    max_samples=int(os.getenv("ONLINE_TRAINING_BATCH", "6")),
                )
            )
            stats.update(
                apply_online_imitation_updates(
                    self._imitation_policy,
                    self._imitation_buffer,
                    max_samples=int(os.getenv("ONLINE_IMITATION_BATCH", "16")),
                    min_samples=int(os.getenv("ONLINE_IMITATION_MIN_SAMPLES", "4")),
                )
            )
            if getattr(self, "_learning_guard", None) is not None:
                try:
                    stats.update(
                        self._learning_guard.evaluate(
                            stats=stats,
                            performance_monitor=self._performance_monitor,
                            predictive_model=self._predictive_model,
                            imitation_policy=self._imitation_policy,
                            learning_manager=self._learning_manager,
                            event_bus=self._event_bus,
                        )
                    )
                except Exception:
                    logger.debug("Learning safety guard failed.", exc_info=True)
        if self._self_improvement is not None:
            try:
                stats.update(
                    self._self_improvement.run_next(
                        event_bus=self._event_bus,
                        retrain_callback=self._schedule_retrain,
                        knowledge_pipeline=getattr(self, "_knowledge_pipeline", None),
                        memory_router=self._memory_router,
                        imitation_policy=self._imitation_policy,
                        predictive_model=self._predictive_model,
                        runtime_config=self._config,
                        now=time.time(),
                    )
                )
            except Exception:
                logger.debug("Self-improvement step failed.", exc_info=True)
        stats.update(self._dispatch_internal_tasks())
        # Placeholder hooks for future trainers; currently just mark metrics.
        try:
            if self._performance_monitor is not None:
                self._performance_monitor.log_snapshot({"learning_cycle": 1.0, **{k: float(v) for k, v in stats.items()}})
        except Exception:
            pass
        if self._adaptive_controller is not None:
            try:
                self._adaptive_controller.record_extra_metrics({"learning_cycle": 1.0})
            except Exception:
                pass
        return stats

    def _consolidate_memories(self) -> Dict[str, Any]:
        """Promote frequently used episodic entries into long-term semantic memory."""

        results: Dict[str, Any] = {}
        try:
            promoted = self._memory_router.review(
                usage_threshold=2,
                max_age=1800.0,
                min_usage=1,
            )
            results["episodic_promoted"] = float(len(promoted))
            stats = self._memory_router.stats()
            results["episodic_total"] = float(stats.get("total_entries", 0))
            results["episodic_promoted_total"] = float(stats.get("promoted", 0))
            try:
                self._knowledge_manager.wait_idle(timeout=1.0)
            except Exception:
                pass
        except Exception:
            logger.debug("Memory consolidation failed.", exc_info=True)
        return results

    def _maybe_queue_cross_domain_self_test(self) -> Dict[str, Any]:
        """Publish a cross-domain self-check goal to exercise transfer."""

        results: Dict[str, Any] = {}
        now = time.time()
        if self._cross_test_interval <= 0 or (now - self._last_cross_test_ts) < self._cross_test_interval:
            return results

        domains: set[str] = set()
        if self._last_perception_event:
            domains.update(self._extract_domains([self._last_perception_event]))
        if self._imitation_buffer:
            for sample in list(self._imitation_buffer)[-5:]:
                domains.update(self._extract_domains([sample.get("state"), sample.get("metadata")]))
        if not domains:
            return results

        domain = sorted(domains)[0]
        goal_text = f"Cross-domain self-check: rehearse task in domain '{domain}' with slight variation"
        self._event_bus.publish("planner.plan_ready", {"goal": goal_text, "tasks": [], "source": "cross_domain_test"})
        self._last_cross_test_ts = now
        results["cross_domain_goal"] = 1.0
        return results

    def _run_cross_domain_evaluator(self) -> Dict[str, Any]:
        """Run optional cross-domain benchmark to measure transfer performance."""

        results: Dict[str, Any] = {}
        if self._cross_domain_benchmark is None:
            return results
        try:
            outcomes = self._cross_domain_benchmark.run()
            if outcomes:
                success_rate = sum(1 for o in outcomes if getattr(o, "success", False)) / len(outcomes)
                results["cross_domain_success"] = float(success_rate)
                if self._adaptive_controller is not None:
                    self._adaptive_controller.record_extra_metrics({"cross_domain_success": success_rate})
                if self._performance_monitor is not None:
                    self._performance_monitor.log_snapshot({"cross_domain_success": success_rate})
                self._update_visualizer_metrics({"cross_domain_success": success_rate})
        except Exception:
            logger.debug("Cross-domain benchmark failed.", exc_info=True)
        return results

    def _dispatch_internal_tasks(self) -> Dict[str, Any]:
        """Schedule and run low-priority maintenance tasks when conditions allow."""

        results: Dict[str, Any] = {}
        now = time.time()
        if (now - self._last_internal_task_ts) < self._internal_task_interval:
            return results
        self._last_internal_task_ts = now

        # Seed daily/periodic maintenance tasks.
        self._internal_tasks.append(
            {"type": "memory_backup", "action": self._memory_backup, "priority": "low"}
        )
        self._internal_tasks.append(
            {"type": "log_compaction", "action": self._rotate_logs, "priority": "low"}
        )
        self._internal_tasks.append(
            {"type": "vector_cache_snapshot", "action": self._snapshot_hot_vectors, "priority": "low"}
        )
        self._internal_tasks.append(
            {"type": "checkpoint", "action": self._checkpoint_state, "priority": "low"}
        )

        # Execute one internal task per learning cycle to avoid contention.
        if self._internal_tasks:
            task = self._internal_tasks.popleft()
            try:
                action = task.get("action")
                if callable(action):
                    action()
                results[f"internal_{task.get('type', 'task')}"] = 1.0
            except Exception:
                logger.debug("Internal task failed: %s", task, exc_info=True)
        return results

    def _memory_backup(self) -> None:
        """Persist key runtime state for recovery."""

        try:
            stats = self._memory_router.stats()
            snapshot_dir = Path(os.getenv("BACKUP_DIR", "snapshots"))
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            state = {
                "memory_stats": stats,
                "tasks": {
                    "backlog": self._scheduler_backlog + self._manager_backlog,
                    "count": self._task_count,
                },
            }
            with open(snapshot_dir / f"state_{ts}.json", "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            logger.debug("Memory backup failed.", exc_info=True)

    def _rotate_logs(self) -> None:
        """Simple log compaction: remove oldest log files beyond retention."""

        try:
            log_dir = Path(os.getenv("LOG_DIR", "logs"))
            retention = int(os.getenv("LOG_RETENTION", "10"))
            if not log_dir.exists():
                return
            log_files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
            if len(log_files) > retention:
                for old in log_files[: len(log_files) - retention]:
                    try:
                        old.unlink()
                    except Exception:
                        logger.debug("Failed to remove log file %s", old, exc_info=True)
        except Exception:
            logger.debug("Log rotation failed.", exc_info=True)

    def _snapshot_hot_vectors(self) -> None:
        """Persist a small cache of hot embeddings to disk for faster warm start."""

        try:
            cache = getattr(self._memory_router, "_embed_cache", None)
            if not isinstance(cache, dict) or not cache:
                return
            snapshot_dir = Path(os.getenv("BACKUP_DIR", "snapshots"))
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            out = snapshot_dir / f"embeddings_{ts}.json"
            head = list(cache.items())[-50:] if len(cache) > 50 else list(cache.items())
            with open(out, "w", encoding="utf-8") as f:
                json.dump(head, f, ensure_ascii=False)
        except Exception:
            logger.debug("Embedding cache snapshot failed.", exc_info=True)

    def _checkpoint_state(self, *, reason: str = "scheduled") -> None:
        """Persist task queues, memory stats, and recent working memory."""

        now = time.time()
        if (now - self._last_checkpoint_ts) < self._checkpoint_interval and reason == "scheduled":
            return
        self._last_checkpoint_ts = now
        try:
            snapshot_dir = Path(os.getenv("BACKUP_DIR", "snapshots"))
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            ts = int(now)
            state = {
                "reason": reason,
                "time": ts,
                "memory_stats": self._memory_router.stats(),
                "tasks": {
                    "external_backlog": self._scheduler_backlog,
                    "manager_backlog": self._manager_backlog,
                    "count": self._task_count,
                },
                "working_memory": list(self._working_memory)[-10:],
                "imitation_buffer": list(self._imitation_buffer)[-10:],
            }
            path = snapshot_dir / f"checkpoint_{ts}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception:
            logger.debug("Checkpoint failed.", exc_info=True)

    def _load_latest_checkpoint(self) -> None:
        """Load most recent checkpoint to restore state after restart."""

        if self._last_checkpoint_loaded:
            return
        snapshot_dir = Path(os.getenv("BACKUP_DIR", "snapshots"))
        if not snapshot_dir.exists():
            return
        checkpoints = sorted(snapshot_dir.glob("checkpoint_*.json"), key=lambda p: p.stat().st_mtime)
        if not checkpoints:
            return
        latest = checkpoints[-1]
        try:
            with open(latest, "r", encoding="utf-8") as f:
                state = json.load(f)
            tasks = state.get("tasks", {})
            self._scheduler_backlog = int(tasks.get("external_backlog", 0))
            self._manager_backlog = int(tasks.get("manager_backlog", 0))
            self._task_count = int(tasks.get("count", 0))
            wm = state.get("working_memory", [])
            if isinstance(wm, list):
                for item in wm:
                    if isinstance(item, dict):
                        self._working_memory.append(item)
            demos = state.get("imitation_buffer", [])
            if isinstance(demos, list):
                for item in demos:
                    if isinstance(item, dict):
                        self._imitation_buffer.append(item)
            self._last_checkpoint_loaded = True
            logger.info("Loaded checkpoint from %s", latest)
        except Exception:
            logger.debug("Failed to load checkpoint %s", latest, exc_info=True)

    def _record_working_memory(self, entry: Dict[str, Any]) -> None:
        """Maintain a short-lived working memory buffer."""

        ts = entry.get("timestamp", time.time())
        entry.setdefault("timestamp", ts)
        self._prune_working_memory(ts)
        self._working_memory.append(entry)
        if self._adaptive_controller is not None:
            try:
                self._adaptive_controller.record_extra_metrics({"working_memory_size": float(len(self._working_memory))})
            except Exception:
                pass

    def _prune_working_memory(self, now: float | None = None) -> None:
        now = now or time.time()
        if self._working_memory_ttl <= 0:
            return
        cutoff = now - self._working_memory_ttl
        while self._working_memory and self._working_memory[0].get("timestamp", now) < cutoff:
            self._working_memory.popleft()

    def _store_episodic_experience(
        self,
        *,
        perception: Mapping[str, Any] | None,
        action: Any,
        reward: Any,
        status: Any,
    ) -> None:
        """Persist an episodic trace to short-term memory for later consolidation."""

        if self._memory_router is None or action is None:
            return
        summary_parts = [
            f"action={action}",
            f"status={status}",
        ]
        if reward is not None:
            summary_parts.append(f"reward={reward}")
        if perception:
            keys = list(perception.keys())
            if keys:
                summary_parts.append(f"perception={keys[:5]}")
            ctx = perception.get("context") if isinstance(perception, dict) else None
            if isinstance(ctx, dict) and ctx:
                summary_parts.append(f"context_keys={list(ctx.keys())[:5]}")
        text = "; ".join(summary_parts)
        metadata = {
            "reward": reward,
            "status": status,
            "time": time.time(),
            "source": "episodic",
        }
        try:
            self._memory_router.add_observation(text, source="episodic", metadata=metadata)
        except Exception:
            logger.debug("Failed to store episodic memory", exc_info=True)

    def _reflect_failure(self, event: Mapping[str, Any], *, action: Any, reward: float | None) -> None:
        """Record failure reflections and propose alternative paths."""

        reason = event.get("error_reason") or event.get("error") or event.get("status")
        summary = f"failure action={action} reason={reason} reward={reward}"
        alternatives = self._suggest_alternatives(action)
        meta = {
            "reason": reason,
            "reward": reward,
            "alternatives": alternatives,
            "timestamp": time.time(),
        }
        try:
            self._memory_router.add_observation(summary, source="failure", metadata=meta)
        except Exception:
            logger.debug("Failed to store failure reflection", exc_info=True)
        payload = {"failures": 1.0}
        try:
            self._adaptive_controller.record_extra_metrics(payload)
        except Exception:
            pass
        if self._performance_monitor is not None:
            try:
                self._performance_monitor.log_snapshot(payload)
            except Exception:
                pass
        self._update_visualizer_metrics(payload)
        self._record_working_memory({"type": "failure", "action": action, "reason": reason, "timestamp": time.time()})
        if alternatives:
            goal_text = f"Replan after failure: consider {', '.join(alternatives)}"
            self._event_bus.publish("planner.plan_ready", {"goal": goal_text, "tasks": [], "source": "failure_reflection"})

    def _suggest_alternatives(self, failed_action: Any) -> list[str]:
        """Suggest alternative actions based on imitation buffer heuristics."""

        alts: list[str] = []
        if self._imitation_policy is not None and self._last_perception_event:
            try:
                suggestions = self._imitation_policy.suggest_actions(
                    self._last_perception_event,
                    top_k=6,
                    exclude=[str(failed_action)] if failed_action else (),
                )
                for action in suggestions:
                    if action:
                        alts.append(str(action))
                    if len(alts) >= 3:
                        break
            except Exception:
                pass
        if alts:
            return alts
        for sample in reversed(self._imitation_buffer):
            act = sample.get("action")
            if act and act != failed_action:
                alts.append(str(act))
            if len(alts) >= 3:
                break
        # Fallback generic alternative
        if not alts and failed_action:
            alts.append(f"avoid {failed_action}")
        return alts

    def _assess_knowledge_gaps(self, domains: set[str]) -> set[str]:
        """Compare task domains against self-model capabilities to find gaps."""

        if not domains:
            return set()
        try:
            capabilities = getattr(self._self_model, "capabilities", {}) or {}
        except Exception:
            capabilities = {}
        gaps: set[str] = set()
        for domain in domains:
            conf = float(capabilities.get(domain, 0.0)) if isinstance(capabilities, dict) else 0.0
            if conf < self._capability_gap_threshold:
                gaps.add(domain)
        return gaps

    @staticmethod
    def _extract_domains(texts: list[Any]) -> set[str]:
        domains: set[str] = set()
        for text in texts:
            if not text:
                continue
            if isinstance(text, dict):
                candidates = []
                for key in ("domains", "tags", "skills", "categories"):
                    val = text.get(key)
                    if isinstance(val, (list, tuple, set)):
                        candidates.extend(val)
                for key in ("topic", "intent", "summary", "name", "category"):
                    val = text.get(key)
                    if isinstance(val, str):
                        candidates.append(val)
                meta = text.get("metadata")
                if isinstance(meta, dict):
                    for key in ("domain", "domains", "tags", "skills", "topics"):
                        val = meta.get(key)
                        if isinstance(val, (list, tuple, set)):
                            candidates.extend(val)
                        elif isinstance(val, str):
                            candidates.append(val)
                for cand in candidates:
                    for token in str(cand).replace("_", " ").split():
                        token = token.strip().lower()
                        if len(token) >= 3:
                            domains.add(token)
                continue
            for token in str(text).replace("_", " ").split():
                token = token.strip().lower()
                if len(token) >= 3:
                    domains.add(token)
        return domains

    # ------------------------------------------------------------------ #
    def _meta_adjust_parameters(self, payload: Mapping[str, Any]) -> None:
        """Lightweight meta-parameter tuning hook."""

        reason = str(payload.get("reason", ""))
        suggested = payload.get("suggested") or {}
        adjustments = payload.get("adjustments") or {}
        exploration_hint = suggested.get("exploration")
        if exploration_hint and hasattr(self._planner, "temperature"):
            try:
                delta = -0.05 if exploration_hint == "decrease" else 0.05
                temp = float(getattr(self._planner, "temperature", 1.0))
                setattr(self._planner, "temperature", max(0.05, temp + delta))
            except Exception:
                pass
        if exploration_hint and self._imitation_policy is not None:
            with self._learning_model_lock:
                try:
                    cfg = getattr(self._imitation_policy, "config", None)
                    if cfg is not None and hasattr(cfg, "inference_uniform_mix"):
                        delta = -0.02 if exploration_hint == "decrease" else 0.02
                        mix = float(getattr(cfg, "inference_uniform_mix", 0.0))
                        setattr(cfg, "inference_uniform_mix", min(1.0, max(0.0, mix + delta)))
                except Exception:
                    pass
        if hasattr(self._config, "goal_frequency") and exploration_hint:
            try:
                delta = -0.1 if exploration_hint == "decrease" else 0.1
                freq = float(getattr(self._config, "goal_frequency", 1.0))
                self._config.goal_frequency = max(0.1, freq + delta)
            except Exception:
                pass
        exp_delta = adjustments.get("exploration_rate")
        if exp_delta and hasattr(self._planner, "temperature"):
            try:
                temp = float(getattr(self._planner, "temperature", 1.0))
                setattr(self._planner, "temperature", max(0.05, temp + float(exp_delta)))
            except Exception:
                pass
        if exp_delta and self._imitation_policy is not None:
            with self._learning_model_lock:
                try:
                    cfg = getattr(self._imitation_policy, "config", None)
                    if cfg is not None and hasattr(cfg, "inference_uniform_mix"):
                        mix = float(getattr(cfg, "inference_uniform_mix", 0.0))
                        setattr(cfg, "inference_uniform_mix", min(1.0, max(0.0, mix + 0.2 * float(exp_delta))))
                except Exception:
                    pass
        lr_delta = adjustments.get("learning_rate")
        if lr_delta and hasattr(self._planner, "learning_rate"):
            try:
                lr = float(getattr(self._planner, "learning_rate", 0.1))
                setattr(self._planner, "learning_rate", max(1e-4, lr + float(lr_delta)))
            except Exception:
                pass
        if lr_delta and self._predictive_model is not None:
            with self._learning_model_lock:
                try:
                    if hasattr(self._predictive_model, "learning_rates") and hasattr(
                        self._predictive_model, "set_learning_rates"
                    ):
                        current = self._predictive_model.learning_rates()  # type: ignore[call-arg]
                        recon = float(current.get("learning_rate", 0.0) or 0.0) + float(lr_delta)
                        pred = float(current.get("prediction_learning_rate", recon) or recon) + float(lr_delta)
                        self._predictive_model.set_learning_rates(  # type: ignore[call-arg]
                            reconstruction_lr=recon,
                            prediction_lr=pred,
                        )
                except Exception:
                    pass
        if lr_delta and self._imitation_policy is not None:
            with self._learning_model_lock:
                try:
                    cfg = getattr(self._imitation_policy, "config", None)
                    if cfg is not None and hasattr(cfg, "lr"):
                        current = float(getattr(cfg, "lr", 0.01))
                        setattr(cfg, "lr", max(1e-4, current + float(lr_delta)))
                except Exception:
                    pass

    def _schedule_retrain(self, payload: Mapping[str, Any]) -> None:
        """Queue a retraining request for offline processing."""

        self._retrain_requests.append(dict(payload))
        if self._performance_monitor is not None:
            try:
                self._performance_monitor.log_snapshot({"retrain_queue": len(self._retrain_requests)})
            except Exception:
                pass

    def _optimize_resources(self, payload: Mapping[str, Any]) -> None:
        """Apply simple resource optimizations based on feedback."""

        action = payload.get("action")
        if action == "reduce_load":
            listener = getattr(self._goal_generator, "listener", None)
            if listener:
                try:
                    base = getattr(self._goal_generator, "listener_poll_interval", 30.0)
                    listener.set_poll_interval(base * 1.5)
                except Exception:
                    pass
        elif action == "optimize_memory":
            try:
                self._memory_router.shrink(max_entries=128, max_age=1800.0, min_usage=1)
            except Exception:
                pass
        self._record_memory_hits()

    # ------------------------------------------------------------------
    # Blueprint change handling
    # ------------------------------------------------------------------
    def _on_blueprint_change(self, path: Path) -> None:
        name = path.stem.split("_v")[0]
        self._paths[name] = path
        try:
            agent = create_agent_from_blueprint(
                path,
                self._config,
                self._llm_provider,
                self._file_storage,
                world_model=self._world_model,
                conductor=self._conductor,
            )
            previous = self._agents.get(name)
            if previous is not None:
                _shutdown_agent(previous)
                self._conductor.unregister_agent(name)
                self._metrics.unregister(name)
                self._set_state(name, AgentState.TERMINATED)
                action = "restarted"
            else:
                action = "spawned"
            self._agents[name] = agent
            self._metrics.register(name, getattr(agent, "pid", os.getpid()))
            self._resources[name] = {
                "cpu": 0.0,
                "memory": 0.0,
                "cpu_pred": 0.0,
                "memory_pred": 0.0,
                "quota": dict(self._default_quota),
                "last_active": time.time(),
            }
            self._heartbeats[name] = time.time()
            self._set_state(name, AgentState.INITIALIZING)
            self._set_state(name, AgentState.RUNNING)
            if self._scheduler:
                self._scheduler.add_agent(name)
            self._event_bus.publish(
                "agent.lifecycle",
                {"agent": name, "action": action, "path": str(path)},
            )
        except AutoGPTException as exc:
            self._event_bus.publish(
                "agent.lifecycle",
                {
                    "agent": name,
                    "action": "failed",
                    "path": str(path),
                    **log_and_format_exception(exc),
                },
            )
        except Exception as exc:  # pragma: no cover - unexpected
            self._event_bus.publish(
                "agent.lifecycle",
                {
                    "agent": name,
                    "action": "failed",
                    "path": str(path),
                    **log_and_format_exception(exc),
                },
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Stop watching and shut down all managed agents."""
        self._watcher.stop()
        self._resource_stop.set()
        self._resource_thread.join()
        if getattr(self, "_learning_manager", None) is not None:
            try:
                self._learning_manager.close()
            except Exception:
                pass
        if getattr(self, "_self_debug_manager", None) is not None:
            try:
                self._self_debug_manager.close()
            except Exception:
                pass
        if getattr(self, "_self_correction_manager", None) is not None:
            try:
                self._self_correction_manager.close()
            except Exception:
                pass
        if getattr(self, "_self_diagnoser", None) is not None:
            try:
                self._self_diagnoser.close()
            except Exception:
                pass
        if getattr(self, "_automl_manager", None) is not None:
            try:
                self._automl_manager.close()
            except Exception:
                pass
        if getattr(self, "_code_self_repair", None) is not None:
            try:
                self._code_self_repair.close()
            except Exception:
                pass
        if getattr(self, "_fault_recovery", None) is not None:
            try:
                self._fault_recovery.close()
            except Exception:
                pass
        if getattr(self, "_scheduler_control", None) is not None:
            try:
                self._scheduler_control.close()
            except Exception:
                pass
        if getattr(self, "_module_lifecycle", None) is not None:
            try:
                self._module_lifecycle.close()
            except Exception:
                pass
        if self._planning_handle and not self._planning_handle.done():
            self._planning_handle.cancel()
        self._planning_handle = None
        self._planning_inflight.clear()
        adapter = getattr(self, "_task_manager_env_adapter", None)
        if adapter is not None and hasattr(adapter, "stop"):
            try:
                adapter.stop(timeout=2.0)
            except Exception:
                pass
        self._task_manager_env_adapter = None
        if self._task_manager:
            self._task_manager.shutdown()
        if self._adaptive_controller:
            self._adaptive_controller.shutdown()
        self._metrics.stop()
        self._goal_generator.stop()
        if self._resource_scheduler:
            self._resource_scheduler.close()
        for name, agent in list(self._agents.items()):
            for key in getattr(agent, "workspace_keys", (name,)):
                global_workspace.unregister_module(key)
            _shutdown_agent(agent)
            if self._scheduler:
                self._scheduler.remove_agent(name)
            self._metrics.unregister(name)
            self._conductor.unregister_agent(name)
            self._set_state(name, AgentState.TERMINATED)
        self._agents.clear()
        self._resources.clear()
        self._conductor.close()

        if hasattr(self, "_async_loop"):
            try:
                self._async_loop.call_soon_threadsafe(self._async_loop.stop)
            except RuntimeError:
                pass
            if hasattr(self, "_async_loop_thread"):
                self._async_loop_thread.join()
            global_workspace.unregister_module("async.loop")
        if hasattr(self, "_async_executor"):
            self._async_executor.shutdown(wait=False)
            global_workspace.unregister_module("async.executor")

    def _handle_async_exception(
        self,
        loop: asyncio.AbstractEventLoop,
        context: Dict[str, Any],
    ) -> None:
        msg = context.get("message") or "Unhandled exception in async loop"
        exc = context.get("exception")
        if exc:
            logger.debug("%s: %s", msg, exc, exc_info=True)
        else:
            logger.debug("%s: %s", msg, context)

    # ------------------------------------------------------------------
    def pause_agent(self, name: str, reason: str | None = None) -> None:
        """Pause *name* if running and mark it sleeping."""
        agent = self._agents.pop(name, None)
        if agent is not None:
            for key in getattr(agent, "workspace_keys", (name,)):
                global_workspace.unregister_module(key)
            _shutdown_agent(agent)
            self._metrics.unregister(name)
            if self._scheduler:
                self._scheduler.remove_agent(name)
            self._resources.pop(name, None)
            self._conductor.unregister_agent(name)
        # Reset heartbeat so termination is based on sleep timeout
        self._heartbeats[name] = time.time()
        self._set_state(name, AgentState.SLEEPING)
        self._event_bus.publish(
            "agent.lifecycle",
            {"agent": name, "action": "paused", "reason": reason} if reason else {"agent": name, "action": "paused"},
        )

    def resume_agent(self, name: str) -> bool:
        """Resume a previously paused agent."""
        if self._states.get(name) != AgentState.SLEEPING:
            return False
        path = self._paths.get(name)
        if not path:
            return False
        try:
            agent = create_agent_from_blueprint(
                path,
                self._config,
                self._llm_provider,
                self._file_storage,
                world_model=self._world_model,
                conductor=self._conductor,
            )
            self._agents[name] = agent
            self._metrics.register(name, getattr(agent, "pid", os.getpid()))
            self._resources[name] = {
                "cpu": 0.0,
                "memory": 0.0,
                "cpu_pred": 0.0,
                "memory_pred": 0.0,
                "quota": dict(self._default_quota),
                "last_active": time.time(),
            }
            self._heartbeats[name] = time.time()
            if self._scheduler:
                self._scheduler.add_agent(name)
            self._set_state(name, AgentState.RUNNING)
            self._event_bus.publish(
                "agent.lifecycle", {"agent": name, "action": "resumed"}
            )
            return True
        except AutoGPTException as exc:
            self._event_bus.publish(
                "agent.lifecycle",
                {
                    "agent": name,
                    "action": "failed",
                    **log_and_format_exception(exc),
                },
            )
            self._set_state(name, AgentState.TERMINATED)
            return False
        except Exception as exc:  # pragma: no cover - unexpected
            self._event_bus.publish(
                "agent.lifecycle",
                {
                    "agent": name,
                    "action": "failed",
                    **log_and_format_exception(exc),
                },
            )
            self._set_state(name, AgentState.TERMINATED)
            return False

    def terminate_agent(self, name: str, reason: str | None = None) -> None:
        """Completely remove *name* and mark it terminated."""
        agent = self._agents.pop(name, None)
        if agent is not None:
            _shutdown_agent(agent)
        self._conductor.unregister_agent(name)
        self._metrics.unregister(name)
        if self._scheduler:
            self._scheduler.remove_agent(name)
        self._resources.pop(name, None)
        self._paths.pop(name, None)
        self._heartbeats.pop(name, None)
        self._set_state(name, AgentState.TERMINATED)
        self._event_bus.publish(
            "agent.lifecycle",
            {"agent": name, "action": "terminated", "reason": reason}
            if reason
            else {"agent": name, "action": "terminated"},
        )

    # ------------------------------------------------------------------
    async def _on_resource_event(self, event: Dict[str, float]) -> None:
        name = event.get("agent")
        if name not in self._resources:
            return
        data = self._resources[name]
        data["cpu"] = event.get("cpu", 0.0)
        data["memory"] = event.get("memory", 0.0)
        quota = event.get("quota")
        if quota:
            data.setdefault("quota", dict(self._default_quota)).update(quota)
        if "last_action" in event:
            data["last_action"] = event["last_action"]
        if self._scheduler:
            self._scheduler.update_agent(name, data["cpu"], data["memory"])
        if data["cpu"] > 1.0:
            data["last_active"] = time.time()
            self._set_state(name, AgentState.RUNNING)

        env_pred = self._world_model.predict(self._resources)
        metrics, summary = self._self_model.assess_state(
            {"cpu": data.get("cpu", 0.0), "memory": data.get("memory", 0.0)},
            env_pred,
            event.get("last_action", data.get("last_action", "")),
        )
        global_workspace.broadcast("self_model", {"agent": name, "summary": summary})

    def _schedule_planning_task(self) -> None:
        if self._planning_inflight.is_set():
            return
        submitter = getattr(self, "_task_submitter", None)
        submit = getattr(submitter, "submit_task", None) if submitter is not None else None
        if callable(submit):
            handle = submit(
                self._background_plan_idle_goal,
                priority=TaskPriority.HIGH,
                deadline=time.time() + 10.0,
                category="planning",
                name="auto_plan_idle_agents",
                metadata={"reason": "idle_agents"},
            )
        else:
            handle = self._task_manager.submit(
                self._background_plan_idle_goal,
                priority=TaskPriority.HIGH,
                deadline=time.time() + 10.0,
                category="planning",
                name="auto_plan_idle_agents",
                metadata={"reason": "idle_agents"},
            )
        self._planning_handle = handle
        self._planning_inflight.set()

    def _background_plan_idle_goal(self) -> Optional[Dict[str, Any]]:
        goal = self._goal_generator.generate()
        if not goal:
            return None
        tasks = self._planner.decompose(goal, source="auto")
        return {"goal": goal, "tasks": tasks}

    def _consume_planning_result(self) -> None:
        handle = self._planning_handle
        if handle is None or not handle.done():
            return
        self._planning_handle = None
        self._planning_inflight.clear()
        try:
            result = handle.result()
        except Exception:
            logger.debug("Automated planning task failed", exc_info=True)
            return
        if not isinstance(result, dict):
            return
        goal = result.get("goal")
        tasks = result.get("tasks") or []
        if goal:
            self._event_bus.publish(
                "planner.plan_ready",
                {"goal": goal, "tasks": tasks if isinstance(tasks, list) else [], "source": "auto"},
            )

    def _resource_manager(self) -> None:
        idle_timeout = 30.0
        alpha = 0.5
        while not self._resource_stop.wait(1.0):
            now = time.time()
            self._consume_planning_result()

            # Check heartbeat timeouts
            for name in list(self._agents.keys()):
                hb = self._heartbeats.get(name, 0.0)
                if now - hb > self._heartbeat_timeout:
                    self.pause_agent(name, reason="heartbeat_timeout")

            # Predictive resource checks and idleness
            env_pred = self._world_model.predict(self._resources)
            cpu_samples: List[float] = []
            memory_samples: List[float] = []
            for name, data in list(self._resources.items()):
                cpu = data.get("cpu", 0.0)
                mem = data.get("memory", 0.0)
                metrics, summary = self._self_model.assess_state(
                    {"cpu": cpu, "memory": mem},
                    env_pred,
                    data.get("last_action", ""),
                )
                self._memory_router.add_observation(
                    summary,
                    source=f"self_model:{name}",
                    metadata={"agent": name},
                )
                try:
                    resource_latency = float(metrics.get("latency", 0.0) or 0.0)
                    resource_throughput = float(metrics.get("throughput", 0.0) or 0.0)
                    self._adaptive_controller.record_module_metric(
                        "self_model", latency=resource_latency, throughput=resource_throughput
                    )
                except Exception:
                    pass
                listener = getattr(self._goal_generator, "listener", None)
                if listener:
                    try:
                        interval = getattr(listener, "poll_interval", None) or getattr(
                            listener, "listener_poll_interval", 0.0
                        )
                        self._adaptive_controller.record_module_metric(
                            "goal_listener", latency=float(interval)
                        )
                    except Exception:
                        pass
                if self._performance_monitor is not None:
                    try:
                        reward = float(metrics.get("reward", 0.0) or 0.0)
                        success = float(metrics.get("success_rate", 0.0) or 0.0)
                        self._performance_monitor.log_inference(reward)
                        self._performance_monitor.log_training(success)
                    except Exception:
                        pass
                cpu, mem = metrics["cpu"], metrics["memory"]
                cpu_samples.append(float(cpu))
                memory_samples.append(float(mem))
                data["cpu_pred"] = alpha * cpu + (1 - alpha) * data.get("cpu_pred", cpu)
                data["memory_pred"] = alpha * mem + (1 - alpha) * data.get("memory_pred", mem)
                self._event_bus.publish(
                    "agent.self_awareness", {"agent": name, "summary": summary}
                )
                record_memory_hit()
                quota = data.get("quota", self._default_quota)
                if (
                    data["cpu_pred"] > quota.get("cpu", 100.0)
                    or data["memory_pred"] > quota.get("memory", 100.0)
                ):
                    self._event_bus.publish(
                        "agent.resource", {"agent": name, "action": "throttle"}
                    )
                if now - data.get("last_active", now) > idle_timeout:
                    if self._states.get(name) == AgentState.RUNNING:
                        self._set_state(name, AgentState.IDLE)

            self._cleanup_sleeping_agents(now)

            # If no tasks are queued and agents are idle, generate new goals
            if self._task_count == 0 and any(
                state == AgentState.IDLE for state in self._states.values()
            ):
                self._schedule_planning_task()

            # Scale agents based on pending tasks
            if self._task_count > len(self._agents):
                self._wake_sleeping_agents(self._task_count - len(self._agents))
            elif self._task_count < len(self._agents):
                self._release_idle_agents(len(self._agents) - self._task_count)
            self._memory_router.review()
            backlog_level = self._scheduler_backlog + self._manager_backlog
            avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
            avg_memory = (
                sum(memory_samples) / len(memory_samples) if memory_samples else 0.0
            )
            if self._adaptive_controller:
                reward_signal = max(0.0, 100.0 - (avg_cpu + avg_memory))
                self._adaptive_controller.update(
                    avg_cpu,
                    avg_memory,
                    backlog_level,
                    metrics={
                        "backlog": backlog_level,
                        "reward": reward_signal,
                        "avg_cpu": avg_cpu,
                        "avg_memory": avg_memory,
                    },
                )
            try:
                self._event_bus.publish(
                    "learning.tick",
                    {
                        "avg_cpu": avg_cpu,
                        "avg_memory": avg_memory,
                        "backlog": backlog_level,
                        "time": now,
                    },
                )
            except Exception:
                pass
            self._maybe_generate_internal_goal(backlog_level)
            self._evaluate_global_confidence()

    def _cleanup_sleeping_agents(self, now: float) -> None:
        for name, state in list(self._states.items()):
            if state != AgentState.SLEEPING:
                continue
            hb = self._heartbeats.get(name, 0.0)
            if now - hb <= self._sleep_timeout:
                continue
            self.terminate_agent(name, reason="sleep_timeout")
            self._states.pop(name, None)

    def _wake_sleeping_agents(self, count: int) -> None:
        sleepers = [n for n, s in self._states.items() if s == AgentState.SLEEPING]
        for name in sleepers[:count]:
            path = self._paths.get(name)
            if not path:
                continue
            try:
                agent = create_agent_from_blueprint(
                    path,
                    self._config,
                    self._llm_provider,
                    self._file_storage,
                    world_model=self._world_model,
                    conductor=self._conductor,
                )
                self._agents[name] = agent
                self._metrics.register(name, getattr(agent, "pid", os.getpid()))
                self._resources[name] = {
                    "cpu": 0.0,
                    "memory": 0.0,
                    "cpu_pred": 0.0,
                    "memory_pred": 0.0,
                    "quota": dict(self._default_quota),
                    "last_active": time.time(),
                }
                self._heartbeats[name] = time.time()
                if self._scheduler:
                    self._scheduler.add_agent(name)
                self._set_state(name, AgentState.RUNNING)
            except AutoGPTException as err:
                log_and_format_exception(err)
                self._set_state(name, AgentState.TERMINATED)
            except Exception as err:  # pragma: no cover - unexpected
                log_and_format_exception(err)
                self._set_state(name, AgentState.TERMINATED)

    def _release_idle_agents(self, count: int) -> None:
        idle = [n for n, s in self._states.items() if s == AgentState.IDLE]
        for name in idle[:count]:
            self.pause_agent(name)


def _shutdown_agent(agent: Agent) -> None:
    """Attempt to gracefully shut down an agent if it supports it."""
    for method_name in ("shutdown", "stop", "close"):
        method = getattr(agent, method_name, None)
        if callable(method):
            try:
                method()  # type: ignore[misc]
            except Exception:  # pragma: no cover - best effort
                pass
            break


__all__ = ["AgentLifecycleManager"]
