"""Production-grade integration of the cognitive architecture.

This module orchestrates the sensory, cognitive, emotional, conscious and
motor components defined in the surrounding package.  The implementation now
supports stateful streaming inputs, neuromorphic hardware backends, cognitive
policy pluggability and telemetry suitable for deployment scenarios.

The :class:`WholeBrainSimulation` class exposes a :meth:`process_cycle` method
which accepts structured input data and returns a detailed
``BrainCycleResult`` containing perception, emotion, and action intent
snapshots for downstream agents.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from numbers import Real
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from schemas.emotion import EmotionType

from .sensory_cortex import VisualCortex, AuditoryCortex, SomatosensoryCortex
from .motor_cortex import MotorCortex
from .cerebellum import Cerebellum
from .oscillations import NeuralOscillations
from .anatomy import BrainAtlas, BrainFunctionalTopology, ConnectomeMatrix
from .limbic import LimbicSystem
from .state import (
    BrainCycleResult,
    BrainRuntimeConfig,
    CognitiveIntent,
    CuriosityState,
    EmotionSnapshot,
    FeelingSnapshot,
    PerceptionSnapshot,
    PersonalityProfile,
    ThoughtSnapshot,
)
from .consciousness import ConsciousnessModel
from .neuromorphic.spiking_network import SpikingNetworkConfig, NeuromorphicBackend, NeuromorphicRunResult
from .self_learning import SelfLearningBrain
from .meta_learning.coordinator import MetaLearningCoordinator
from .neuromorphic.temporal_encoding import latency_encode, rate_encode, decode_spike_counts
from .perception import SensoryPipeline, EncodedSignal
from modules.perception.semantic_bridge import SemanticBridge

try:  # Optional reasoning stack
    from backend.reasoning.registry import get_causal_reasoner
except Exception:  # pragma: no cover - optional during tests
    get_causal_reasoner = None  # type: ignore

try:  # Optional dependency chain
    from modules.knowledge import KnowledgeFact, LongTermMemoryCoordinator
except Exception:  # pragma: no cover - knowledge stack may be optional
    KnowledgeFact = None  # type: ignore
    LongTermMemoryCoordinator = None  # type: ignore
from .motor.precision import PrecisionMotorSystem
from .motor.actions import MotorExecutionResult, MotorPlan
from modules.environment import get_hardware_registry
from .self_model import SelfModel
from .whole_brain_policy import (
    CognitiveModule,
    CognitivePolicy,
    BanditCognitivePolicy,
    HeuristicCognitivePolicy,
    ProductionCognitivePolicy,
    ReinforcementCognitivePolicy,
    StructuredPlanner,
    default_plan_for_intention,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from modules.learning.continual_learning import ContinualLearningCoordinator


logger = logging.getLogger(__name__)



@dataclass
class WholeBrainSimulation:
    """Container object coordinating all brain subsystems."""

    _CHANNEL_TO_MODULE: ClassVar[Dict[str, str]] = {
        "observe": "visual",
        "approach": "motor",
        "withdraw": "emotion",
        "explore": "precision_motor",
    }
    _MODULATOR_TO_MODULE: ClassVar[Dict[str, str]] = {
        "dopamine": "curiosity",
        "serotonin": "emotion",
        "norepinephrine": "consciousness",
        "acetylcholine": "cognition",
        "oxytocin": "emotion",
        "histamine": "consciousness",
    }

    visual: VisualCortex = field(default_factory=VisualCortex)
    auditory: AuditoryCortex = field(default_factory=AuditoryCortex)
    somatosensory: SomatosensoryCortex = field(default_factory=SomatosensoryCortex)
    cognition: CognitiveModule = field(default_factory=CognitiveModule)
    personality: PersonalityProfile = field(default_factory=PersonalityProfile)
    emotion: LimbicSystem = field(init=False)
    consciousness: ConsciousnessModel = field(default_factory=ConsciousnessModel)
    motor: MotorCortex = field(default_factory=MotorCortex)
    precision_motor: PrecisionMotorSystem = field(default_factory=PrecisionMotorSystem)
    curiosity: CuriosityState = field(default_factory=CuriosityState)
    cerebellum: Cerebellum = field(default_factory=Cerebellum)
    oscillations: NeuralOscillations = field(default_factory=NeuralOscillations)
    config: BrainRuntimeConfig = field(default_factory=BrainRuntimeConfig)
    self_learning: SelfLearningBrain = field(default_factory=SelfLearningBrain)
    meta_learning: MetaLearningCoordinator = field(default_factory=MetaLearningCoordinator)
    perception_pipeline: SensoryPipeline = field(default_factory=SensoryPipeline)
    atlas: BrainAtlas = field(default_factory=BrainAtlas.default)
    connectome_dataset: str = "hcp"
    connectome: ConnectomeMatrix = field(init=False)
    topology: BrainFunctionalTopology = field(init=False)
    neuromorphic: bool = True
    neuromorphic_encoding: str = "rate"
    encoding_steps: int = 5
    encoding_time_scale: float = 1.0
    max_neurons: int = 128
    max_cache_size: int = 8
    cycle_index: int = field(init=False, default=0)
    last_perception: PerceptionSnapshot = field(init=False, default_factory=PerceptionSnapshot)
    last_context: Dict[str, Any] = field(init=False, default_factory=dict)
    last_learning_prediction: Dict[str, float] = field(init=False, default_factory=dict)
    last_decision: Dict[str, Any] = field(init=False, default_factory=dict)
    last_memory_retrieval: Dict[str, Any] = field(init=False, default_factory=dict)
    last_oscillation_state: Dict[str, float] = field(init=False, default_factory=dict)
    last_motor_result: Optional[NeuromorphicRunResult] = field(init=False, default=None)
    last_topology: Dict[str, Any] = field(init=False, default_factory=dict)
    _spiking_cache: OrderedDict[tuple[int, str], NeuromorphicBackend] = field(init=False, default_factory=OrderedDict)
    _motor_backend: Optional[NeuromorphicBackend] = field(init=False, default=None)
    _exploration_goal_queue: deque[tuple[str, Dict[str, Any]]] = field(
        init=False, default_factory=deque
    )
    _scheduled_exploration_goals: Dict[str, str] = field(
        init=False, default_factory=dict
    )
    perception_history: deque[PerceptionSnapshot] = field(
        init=False, default_factory=lambda: deque(maxlen=32)
    )
    decision_history: deque[Dict[str, Any]] = field(
        init=False, default_factory=lambda: deque(maxlen=32)
    )
    telemetry_log: deque[Dict[str, Any]] = field(
        init=False, default_factory=lambda: deque(maxlen=64)
    )
    _stream_state: Dict[str, Any] = field(init=False, default_factory=dict)
    self_model: SelfModel = field(init=False)
    knowledge_base: Any | None = None
    long_term_memory: Any | None = None
    continual_learning: "ContinualLearningCoordinator | None" = None
    _metrics_collector: Any | None = field(init=False, default=None, repr=False)
    _evolution_cognition: Any | None = field(init=False, default=None, repr=False)
    _evolution_architecture: Any | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.personality.clamp()
        capability_defaults = getattr(self.config, "capability_prior", {}) or {}
        if not isinstance(capability_defaults, dict):
            capability_defaults = {}
        self.self_model = SelfModel(capability_defaults=capability_defaults)
        self.emotion = LimbicSystem(self.personality)
        self.topology = BrainFunctionalTopology(self.atlas)
        self.connectome = ConnectomeMatrix.from_atlas(
            self.atlas, dataset=self.connectome_dataset
        )
        self.config.use_neuromorphic = self.neuromorphic
        self.motor.cerebellum = self.cerebellum
        self.motor.precision_system = self.precision_motor
        if hasattr(self.precision_motor, "basal_ganglia"):
            self.motor.basal_ganglia = self.precision_motor.basal_ganglia
        self.motor.spiking_backend = None
        self.last_oscillation_state = {}
        self.last_motor_result = None
        self._motor_backend = None
        self.last_perception = PerceptionSnapshot()
        self.last_context = {}
        self.last_learning_prediction = {}
        self.last_decision = {}
        self.last_memory_retrieval = {}
        self.cycle_index = 0
        self._latest_neuromorphic_capability: Dict[str, Any] = {}
        self._cached_gpu_available: Optional[bool] = None
        history_size = max(4, self.max_cache_size)
        self.perception_history = deque(maxlen=history_size)
        self.decision_history = deque(maxlen=history_size)
        self.telemetry_log = deque(maxlen=max(8, history_size * 2))
        self._stream_state = {}
        if LongTermMemoryCoordinator is not None and self.long_term_memory is None:
            try:
                self.long_term_memory = LongTermMemoryCoordinator()
            except Exception:  # pragma: no cover - coordinator stack optional
                self.long_term_memory = None
                logger.debug(
                    "Long-term memory coordinator initialisation failed.",
                    exc_info=True,
                )
        try:
            self.semantic_bridge = SemanticBridge()
        except Exception:  # pragma: no cover - optional dependency during tests
            self.semantic_bridge = None
            logger.debug("Semantic bridge initialisation failed.", exc_info=True)
        try:
            registry = get_hardware_registry()
            registry.register(f"brain-{id(self)}", {"backend_type": "cpu", "supports_hardware": False})
        except Exception:
            logger.debug("Failed to register brain CPU capability profile.", exc_info=True)

        if self.config.prefer_reinforcement_planner and not isinstance(
            self.cognition.policy, ReinforcementCognitivePolicy
        ):
            self.cognition.set_policy(ReinforcementCognitivePolicy())

        if (
            (self.config.enable_continual_learning or self.config.enable_self_evolution)
            and self.continual_learning is None
        ):
            self._init_continual_learning()

        if self.config.enable_self_evolution and self.continual_learning is not None:
            self._init_self_evolution()

    def _init_continual_learning(self) -> None:
        try:
            from modules.learning.continual_learning import (
                ContinualLearningCoordinator,
                LearningLoopConfig,
            )
            from modules.learning.online_trainers import make_bandit_intention_policy_trainer
            from modules.meta_cognition import MetaCognitionController
        except Exception:  # pragma: no cover - optional subsystem
            logger.debug("Continual learning dependencies are unavailable.", exc_info=True)
            return

        loop_config = LearningLoopConfig(background_interval=self.config.continual_learning_background_interval)
        coordinator = ContinualLearningCoordinator(
            experience_root=self.config.continual_learning_experience_root,
            config=loop_config,
        )
        coordinator.set_policy_trainer(
            make_bandit_intention_policy_trainer(
                self.cognition,
                policy_path=self.config.continual_learning_policy_path,
            )
        )
        if self.config.enable_meta_cognition or self.config.enable_self_evolution:
            try:
                controller = MetaCognitionController(
                    failure_threshold=int(self.config.meta_failure_threshold),
                    low_confidence_threshold=float(self.config.meta_low_confidence_threshold),
                    low_confidence_window=int(self.config.meta_low_confidence_window),
                    knowledge_gap_threshold=int(self.config.meta_knowledge_gap_threshold),
                )
                coordinator.attach_meta_controller(controller)
            except Exception:
                logger.debug("Meta-cognition controller init failed.", exc_info=True)
        if self.long_term_memory is not None:
            knowledge_importer = getattr(self.long_term_memory, "_importer", None)
            vector_store = getattr(self.long_term_memory, "vector_store", None)
            if knowledge_importer is not None and hasattr(knowledge_importer, "ingest_facts"):
                try:
                    coordinator.attach_memory_consolidator_from_components(
                        knowledge_importer=knowledge_importer,
                        vector_store=vector_store if vector_store is not None else None,
                    )
                except Exception:
                    logger.debug("Continual learning memory consolidator attach failed.", exc_info=True)

        # Optional self-monitoring reflection loop (persists to KnowledgeBase).
        try:
            from modules.learning.agent_reflector import AgentReflector, build_reflection_callback

            reflector = AgentReflector.from_env()
            if reflector is not None:
                coordinator.set_reflection_callback(build_reflection_callback(reflector))
        except Exception:  # pragma: no cover - optional subsystem
            logger.debug("Agent reflector attach failed.", exc_info=True)
        try:
            coordinator.start_background_worker()
        except Exception:  # pragma: no cover - defensive
            logger.debug("Failed to start continual learning background worker.", exc_info=True)
        self.continual_learning = coordinator

    def _init_self_evolution(self) -> None:
        """Wire genetic self-evolution into continual learning using long-term feedback."""

        coordinator = self.continual_learning
        if coordinator is None:
            return
        if self._evolution_architecture is not None:
            return
        try:
            from modules.monitoring.collector import RealTimeMetricsCollector
            from modules.monitoring.performance_diagnoser import PerformanceDiagnoser
            from modules.evolution.evolving_cognitive_architecture import (
                EvolvingCognitiveArchitecture,
                GAConfig,
                GeneticAlgorithm,
            )
            from modules.evolution.self_evolving_cognition import SelfEvolvingCognition
            from modules.evolution.self_evolving_ai_architecture import SelfEvolvingAIArchitecture
            from modules.evolution.structural_evolution import StructuralEvolutionManager
            from modules.evolution.strategy_adjuster import StrategyAdjuster
        except Exception:  # pragma: no cover - optional subsystem
            logger.debug("Self-evolution stack is unavailable.", exc_info=True)
            return

        collector = self._metrics_collector
        if collector is None:
            monitor = None
            try:  # optional backend monitoring stack for dashboards
                from backend.monitoring.performance_monitor import PerformanceMonitor
                from backend.monitoring.storage import TimeSeriesStorage

                monitor = PerformanceMonitor(
                    TimeSeriesStorage(),
                    training_accuracy=0.8,
                    degradation_threshold=0.05,
                )
            except Exception:
                monitor = None
            collector = RealTimeMetricsCollector(monitor=monitor)
            self._metrics_collector = collector
        coordinator.set_collector(collector)

        def _fitness_fn(candidate: Dict[str, float]) -> float:
            # Behaviour feedback (long-term).
            events = collector.events() if collector is not None else []
            avg_latency = sum(e.latency for e in events) / len(events) if events else 0.0
            avg_energy = sum(e.energy for e in events) / len(events) if events else 0.0
            avg_throughput = sum(e.throughput for e in events) / len(events) if events else 0.0

            episodes = coordinator.experience_hub.latest(limit=40) if coordinator.experience_hub is not None else []
            if episodes:
                success_rate = sum(1 for e in episodes if getattr(e, "success", False)) / len(episodes)
                avg_reward = sum(float(getattr(e, "total_reward", 0.0) or 0.0) for e in episodes) / len(episodes)
            else:
                success_rate = 0.0
                avg_reward = 0.0

            # Heuristic target derived from observed behaviour: evolve towards a regime
            # that improves success when struggling, and trims overhead when stable.
            desired_exploration = max(0.02, min(0.6, 0.05 + (1.0 - success_rate) * 0.35))
            desired_lr = max(1e-3, min(0.5, 0.03 + avg_throughput * 0.01))
            desired_structured = 1.0 if success_rate < 0.85 else 0.8
            desired_reinforcement = 1.0 if (success_rate > 0.8 and avg_latency < 0.6) else 0.0
            desired_policy_variant = 1.0 if success_rate < 0.75 else 0.0
            desired_planner_steps = 6.0 if success_rate < 0.6 else 4.0
            desired_replay_buffer = 512.0 if success_rate < 0.6 else 256.0
            desired_replay_batch = 32.0 if success_rate < 0.6 else 16.0
            desired_replay_iters = 3.0 if success_rate < 0.6 else 1.0

            targets = {
                "policy_exploration_rate": desired_exploration,
                "policy_learning_rate": desired_lr,
                "planner_structured_flag": desired_structured,
                "planner_reinforcement_flag": desired_reinforcement,
                "cognitive_policy_variant": desired_policy_variant,
                "planner_min_steps": desired_planner_steps,
                "policy_replay_buffer_size": desired_replay_buffer,
                "policy_replay_batch_size": desired_replay_batch,
                "policy_replay_iterations": desired_replay_iters,
                "policy_hidden_dim": 128.0,
                "policy_num_layers": 2.0,
                "module_self_learning_flag": 1.0 if success_rate < 0.9 else 0.7,
                "module_curiosity_feedback_flag": 1.0 if success_rate < 0.9 else 0.8,
                "module_metrics_flag": 1.0,
            }

            def _sq(x: float) -> float:
                return x * x

            alignment = 0.0
            for key, target in targets.items():
                value = float(candidate.get(key, target))
                diff = value - float(target)
                if key in {"policy_replay_buffer_size", "policy_replay_batch_size", "policy_hidden_dim"}:
                    try:
                        diff = math.log2(max(1.0, value)) - math.log2(max(1.0, float(target)))
                    except ValueError:
                        diff = 0.0
                elif key in {"planner_min_steps", "policy_replay_iterations", "policy_num_layers"}:
                    denom = max(1.0, float(target))
                    diff = diff / denom
                elif key == "cognitive_policy_variant":
                    diff = diff / 2.0
                alignment -= _sq(diff)

            # Long-term performance baseline.
            baseline = success_rate * 2.0 + avg_reward * 0.2
            baseline += avg_throughput * 0.05
            baseline -= avg_latency * 0.5
            baseline -= avg_energy * 0.05

            return float(baseline + 0.35 * alignment)

        ga_cfg = GAConfig(
            population_size=int(self.config.evolution_ga_population),
            generations=int(self.config.evolution_ga_generations),
            mutation_rate=float(self.config.evolution_ga_mutation_rate),
            mutation_sigma=float(self.config.evolution_ga_mutation_sigma),
        )

        def _postprocess(candidate: Dict[str, float]) -> None:
            # Keep bounds stable for GA/NAS candidates; final hard clamping
            # still happens in SelfEvolvingAIArchitecture._normalise_architecture.
            candidate["policy_learning_rate"] = float(
                max(1e-4, min(1.0, candidate.get("policy_learning_rate", 0.08)))
            )
            candidate["policy_exploration_rate"] = float(
                max(0.0, min(1.0, candidate.get("policy_exploration_rate", 0.12)))
            )
            candidate["planner_min_steps"] = float(
                max(1, min(16, int(round(candidate.get("planner_min_steps", 4.0)))))
            )
            candidate["policy_replay_buffer_size"] = float(
                max(32, min(4096, int(round(candidate.get("policy_replay_buffer_size", 256.0)))))
            )
            candidate["policy_replay_batch_size"] = float(
                max(1, min(256, int(round(candidate.get("policy_replay_batch_size", 16.0)))))
            )
            candidate["policy_replay_iterations"] = float(
                max(1, min(12, int(round(candidate.get("policy_replay_iterations", 1.0)))))
            )
            candidate["policy_hidden_dim"] = float(
                max(8, min(2048, int(round(candidate.get("policy_hidden_dim", 128.0)))))
            )
            candidate["policy_num_layers"] = float(
                max(1, min(8, int(round(candidate.get("policy_num_layers", 2.0)))))
            )
            candidate["cognitive_policy_variant"] = float(
                max(0, min(2, int(round(candidate.get("cognitive_policy_variant", 0.0)))))
            )
            for flag in (
                "planner_structured_flag",
                "planner_reinforcement_flag",
                "module_self_learning_flag",
                "module_curiosity_feedback_flag",
                "module_metrics_flag",
            ):
                if flag in candidate:
                    candidate[flag] = 1.0 if float(candidate.get(flag, 0.0)) >= 0.5 else 0.0

        ga = GeneticAlgorithm(_fitness_fn, config=ga_cfg, post_mutation=_postprocess)
        nas_controller = None
        if os.environ.get("BSS_META_NAS_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}:
            try:
                from modules.evolution.meta_nas import MetaNASController
            except Exception:
                MetaNASController = None  # type: ignore[assignment]

            if MetaNASController is not None:
                try:
                    nas_controller = MetaNASController(
                        seed=int(os.environ.get("BSS_META_NAS_SEED", "0") or 0),
                        population_size=int(self.config.evolution_ga_population),
                        generations=int(self.config.evolution_ga_generations),
                        postprocess=_postprocess,
                        reward_baseline="best",
                    )
                except Exception:
                    nas_controller = None

        evolver = EvolvingCognitiveArchitecture(_fitness_fn, ga, nas_controller=nas_controller)

        active_policy = getattr(self.cognition, "policy", None)
        policy_variant = 0.0
        if isinstance(active_policy, BanditCognitivePolicy):
            policy_variant = 2.0
        elif isinstance(active_policy, ReinforcementCognitivePolicy):
            policy_variant = 1.0
        elif self.config.prefer_reinforcement_planner:
            policy_variant = 1.0

        planner = getattr(active_policy, "planner", None) if active_policy is not None else None
        planner_min_steps = float(getattr(planner, "min_steps", 4) if planner is not None else 4.0)
        replay_buffer = getattr(active_policy, "_experience_buffer", None) if active_policy is not None else None
        replay_buffer_size = float(getattr(replay_buffer, "maxlen", 256) or 256)
        replay_batch_size = float(getattr(active_policy, "_replay_batch_size", 16) if active_policy is not None else 16.0)
        replay_iterations = float(getattr(active_policy, "_replay_iterations", 1) if active_policy is not None else 1.0)

        initial_arch = {
            "policy_learning_rate": 0.08,
            "policy_exploration_rate": 0.12,
            "cognitive_policy_variant": policy_variant,
            "planner_min_steps": planner_min_steps,
            "policy_replay_buffer_size": replay_buffer_size,
            "policy_replay_batch_size": replay_batch_size,
            "policy_replay_iterations": replay_iterations,
            "policy_hidden_dim": 128.0,
            "policy_num_layers": 2.0,
            "planner_structured_flag": 1.0 if self.config.prefer_structured_planner else 0.0,
            "planner_reinforcement_flag": 1.0 if self.config.prefer_reinforcement_planner else 0.0,
            "module_self_learning_flag": 1.0 if self.config.enable_self_learning else 0.0,
            "module_curiosity_feedback_flag": 1.0 if self.config.enable_curiosity_feedback else 0.0,
            "module_metrics_flag": 1.0 if self.config.metrics_enabled else 0.0,
        }

        cognition = SelfEvolvingCognition(initial_architecture=dict(initial_arch), evolver=evolver, collector=collector)
        architecture = SelfEvolvingAIArchitecture(
            initial_architecture=dict(initial_arch),
            evolver=evolver,
            collector=collector,
            cognition=cognition,
            curiosity_state=self.curiosity,
            brain_config=self.config,
            policy_module=self.cognition,
            continual_learning=coordinator,
        )
        self._evolution_cognition = cognition
        self._evolution_architecture = architecture

        # Optional structural evolution: module gating / rewiring proposals (no expander attached here).
        try:
            structural = StructuralEvolutionManager(architecture)
            coordinator.attach_structural_manager(structural)
        except Exception:
            logger.debug("Structural evolution init failed.", exc_info=True)

        # Diagnostics / rollback loop.
        try:
            diagnoser = PerformanceDiagnoser()
            adjuster = StrategyAdjuster()

            class _EvolutionProxy:
                def __init__(self, cognition_obj: Any, arch_obj: Any) -> None:
                    self.cognition = cognition_obj
                    self._arch = arch_obj

                def rollback(self, version: int) -> Dict[str, float]:
                    restored = self._arch.rollback(int(version))
                    self.cognition.architecture = restored
                    return restored

            coordinator.attach_diagnoser(
                diagnoser,
                strategy_adjuster=adjuster,
                evolution_engine=_EvolutionProxy(cognition, architecture),
            )
        except Exception:
            logger.debug("Evolution diagnoser init failed.", exc_info=True)

    def attach_knowledge_base(self, knowledge_base: Any) -> None:
        """Attach an external knowledge base interface."""

        self.knowledge_base = knowledge_base
        if self.continual_learning is not None and hasattr(self.continual_learning, "attach_knowledge_importer"):
            if hasattr(knowledge_base, "ingest_facts"):
                try:
                    self.continual_learning.attach_knowledge_importer(knowledge_base)  # type: ignore[arg-type]
                except Exception:
                    logger.debug("Continual learning knowledge importer attach failed.", exc_info=True)

    def shutdown(self, *, wait: bool = False) -> None:
        """Stop background learning threads (best-effort)."""

        if self.continual_learning is not None:
            try:
                self.continual_learning.shutdown(wait=wait)
            except Exception:
                logger.debug("Continual learning shutdown failed.", exc_info=True)

    def submit_human_feedback(
        self,
        *,
        task_id: str,
        prompt: str,
        agent_response: str,
        correct_response: str | None = None,
        rating: float | None = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Forward supervised feedback into the continual learning/meta-cognition stack."""

        if self.continual_learning is None:
            if self.config.enable_continual_learning or self.config.enable_self_evolution:
                self._init_continual_learning()
            if self.config.enable_self_evolution and self.continual_learning is not None:
                self._init_self_evolution()
        if self.continual_learning is None or not hasattr(self.continual_learning, "register_human_feedback"):
            raise RuntimeError("Continual learning is not enabled; cannot accept human feedback.")
        return self.continual_learning.register_human_feedback(
            task_id=task_id,
            prompt=prompt,
            agent_response=agent_response,
            correct_response=correct_response,
            rating=rating,
            metadata=metadata,
        )

    def _summarise_perception(self, perception: PerceptionSnapshot) -> Dict[str, float]:
        return self.cognition._summarise_perception(perception)

    def _compose_module_activity(
        self,
        perception_summary: Mapping[str, float],
        emotion: EmotionSnapshot,
        curiosity: CuriosityState,
        decision: Mapping[str, Any],
        oscillation_state: Optional[Mapping[str, float]] = None,
        motor_result: Optional[NeuromorphicRunResult] = None,
    ) -> Dict[str, float]:
        def _normalise(value: Any) -> float:
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                return 0.0

        activity: Dict[str, float] = {module: 0.0 for module in self.topology.module_to_regions}

        activity["visual"] = _normalise(
            perception_summary.get("vision")
            or perception_summary.get("visual")
            or perception_summary.get("image")
        )
        activity["auditory"] = _normalise(
            perception_summary.get("auditory") or perception_summary.get("audio")
        )
        activity["somatosensory"] = _normalise(
            perception_summary.get("somatosensory") or perception_summary.get("touch")
        )
        activity["emotion"] = _normalise(getattr(emotion, "intensity", 0.0))
        activity["curiosity"] = _normalise(getattr(curiosity, "drive", 0.0))

        confidence = decision.get("confidence", 0.0)
        plan_length = len(decision.get("plan", []))
        activity["cognition"] = max(_normalise(confidence), _normalise(plan_length / 5.0))

        if oscillation_state:
            numeric_values = [
                float(v) for v in oscillation_state.values() if isinstance(v, (int, float))
            ]
            avg = sum(numeric_values) / len(numeric_values) if numeric_values else 0.0
            activity["consciousness"] = _normalise(avg)
        else:
            activity["consciousness"] = max(activity.get("consciousness", 0.0), 0.3)

        weights = decision.get("weights", {})
        motor_drive = float(weights.get("approach", 0.0) + weights.get("withdraw", 0.0)) / 2.0
        activity["motor"] = _normalise(motor_drive)
        activity["precision_motor"] = _normalise(weights.get("explore", 0.0) * 0.5)

        if motor_result is not None:
            if motor_result.spike_counts:
                spike_avg = sum(motor_result.spike_counts) / max(
                    1, len(motor_result.spike_counts) * float(self.max_neurons)
                )
                activity["motor"] = max(activity["motor"], _normalise(spike_avg))
            if motor_result.average_rate:
                rate_avg = sum(motor_result.average_rate) / len(motor_result.average_rate)
                activity["precision_motor"] = max(
                    activity["precision_motor"], _normalise(rate_avg)
                )

        activity.setdefault("consciousness", _normalise(confidence))
        for module, value in list(activity.items()):
            activity[module] = _normalise(value)
        return activity

    @staticmethod
    def _optional_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _normalise_fact_payload(self, payload: Any) -> Optional[KnowledgeFact]:
        if KnowledgeFact is None:
            return None
        if isinstance(payload, KnowledgeFact):
            return payload
        if not isinstance(payload, dict):
            return None
        subject = str(payload.get("subject") or "").strip()
        predicate = str(payload.get("predicate") or "").strip()
        obj = str(payload.get("object") or payload.get("obj") or "").strip()
        if not subject or not predicate or not obj:
            return None
        return KnowledgeFact(
            subject=subject,
            predicate=predicate,
            obj=obj,
            subject_id=self._optional_str(payload.get("subject_id")),
            object_id=self._optional_str(payload.get("object_id")),
            subject_description=self._optional_str(payload.get("subject_description")),
            object_description=self._optional_str(payload.get("object_description")),
            metadata=dict(payload.get("metadata") or {}),
            confidence=self._optional_float(payload.get("confidence")),
            source=self._optional_str(payload.get("source")),
            context=self._optional_str(payload.get("context")),
            timestamp=self._optional_float(payload.get("timestamp")),
        )

    def _normalise_causal_relation(self, relation: Any) -> Optional[Dict[str, Any]]:
        cause: str = ""
        effect: str = ""
        weight: Optional[float] = None
        raw_weight: Any = None
        metadata: Dict[str, Any] = {}
        description: Optional[str] = None
        if relation is None:
            return None
        if hasattr(relation, "cause") or hasattr(relation, "effect"):
            cause = str(getattr(relation, "cause", "") or "").strip()
            effect = str(getattr(relation, "effect", "") or "").strip()
            raw_weight = getattr(relation, "weight", None)
            description = getattr(relation, "description", None)
            metadata_payload = getattr(relation, "metadata", None)
            if isinstance(metadata_payload, dict):
                metadata = dict(metadata_payload)
        elif isinstance(relation, dict):
            cause = str(relation.get("cause") or relation.get("source") or "").strip()
            effect = str(relation.get("effect") or relation.get("target") or "").strip()
            raw_weight = relation.get("weight")
            description = relation.get("description") or relation.get("summary")
            metadata_payload = relation.get("metadata")
            if isinstance(metadata_payload, dict):
                metadata = dict(metadata_payload)
        elif isinstance(relation, (tuple, list)) and relation:
            cause = str(relation[0] if len(relation) > 0 else "").strip()
            effect = str(relation[1] if len(relation) > 1 else "").strip()
            raw_weight = relation[2] if len(relation) > 2 else None
        else:
            return None
        if not cause and not effect:
            return None
        try:
            weight = float(raw_weight) if raw_weight is not None else None  # type: ignore[arg-type]
        except (TypeError, ValueError):
            weight = None
        payload: Dict[str, Any] = {}
        if cause:
            payload["cause"] = cause
        if effect:
            payload["effect"] = effect
        if weight is not None:
            payload["weight"] = weight
        if description:
            payload["description"] = str(description)
        if metadata:
            payload["metadata"] = {
                key: value for idx, (key, value) in enumerate(metadata.items()) if idx < 4
            }
        return payload or None

    def _inject_causal_context(
        self,
        knowledge_context: Mapping[str, Any],
        cognitive_context: Dict[str, Any],
        *,
        knowledge_query: str = "",
    ) -> None:
        relations_raw = knowledge_context.get("causal_relations") if isinstance(knowledge_context, Mapping) else None
        if not relations_raw:
            return
        normalised: List[Dict[str, Any]] = []
        for relation in relations_raw:  # type: ignore[assignment]
            payload = self._normalise_causal_relation(relation)
            if payload:
                normalised.append(payload)
            if len(normalised) >= 6:
                break
        if not normalised:
            return
        if isinstance(knowledge_context, dict):
            knowledge_context["causal_relations"] = normalised
        cognitive_context["causal_relations"] = list(normalised)
        if knowledge_query:
            cognitive_context.setdefault("causal_query", knowledge_query)
        focus_relation = next((item for item in normalised if item.get("effect")), normalised[0])
        focus_target = focus_relation.get("effect") or focus_relation.get("cause")
        if isinstance(focus_target, str) and focus_target.strip():
            cognitive_context.setdefault("causal_focus", focus_target.strip())
        reasoner = None
        if get_causal_reasoner is not None:
            try:
                reasoner = get_causal_reasoner()
            except Exception:  # pragma: no cover - optional stack may be absent
                reasoner = None
        if reasoner is None:
            return
        causal_paths: List[Dict[str, Any]] = []
        for relation in normalised[:2]:
            cause = relation.get("cause")
            effect = relation.get("effect")
            if not (isinstance(cause, str) and isinstance(effect, str) and cause and effect):
                continue
            try:
                exists, path = reasoner.check_causality(cause, effect)
            except Exception:  # pragma: no cover - defensive guard
                continue
            if not exists:
                continue
            try:
                path_list = list(path)
            except TypeError:  # pragma: no cover - defensive
                path_list = []
            if not path_list:
                continue
            collapsed = [str(node) for node in path_list[:8]]
            causal_paths.append({"cause": cause, "effect": effect, "path": collapsed})
        if causal_paths:
            cognitive_context["causal_paths"] = causal_paths
            if isinstance(knowledge_context, dict):
                knowledge_context["causal_paths"] = causal_paths

    def _record_long_term_facts(
        self,
        facts: Iterable[Any],
        *,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if (
            self.long_term_memory is None
            or LongTermMemoryCoordinator is None
            or KnowledgeFact is None
        ):
            return
        normalised: List[KnowledgeFact] = []
        for payload in facts:
            fact = self._normalise_fact_payload(payload)
            if fact is not None:
                normalised.append(fact)
        if not normalised:
            return
        base_metadata: Dict[str, Any] = {
            "source": source,
            "cycle_index": self.cycle_index,
        }
        if metadata:
            base_metadata.update(metadata)
        try:
            self.long_term_memory.record_facts(
                normalised,
                base_metadata=base_metadata,
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.debug(
                "Failed to persist knowledge facts to long-term memory.",
                exc_info=True,
            )

    def _augment_context_with_long_term_memory(
        self,
        context: Dict[str, Any],
        perception_summary: Mapping[str, float],
        perception_snapshot: PerceptionSnapshot,
        *,
        intent_hint: Optional[str] = None,
        fallback_text: str = "",
    ) -> None:
        if self.long_term_memory is None or LongTermMemoryCoordinator is None:
            return
        query_candidates: List[str] = []
        if intent_hint:
            hint_text = str(intent_hint).strip()
            if hint_text:
                query_candidates.append(hint_text)
        fallback_text = (fallback_text or "").strip()
        if fallback_text:
            query_candidates.append(fallback_text)
        if perception_summary:
            ranked_modalities = sorted(
                perception_summary.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            modality_tokens = [
                name for name, score in ranked_modalities if score > 0.0
            ]
            if modality_tokens:
                query_candidates.append(" ".join(modality_tokens[:4]))
        query_text = next((candidate for candidate in query_candidates if candidate), "")
        if not query_text:
            self.last_memory_retrieval = {}
            return
        try:
            records = self.long_term_memory.query_similar(query_text, top_k=5)
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Long-term memory similarity lookup failed.", exc_info=True)
            records = []
        record_payload: List[Dict[str, Any]] = [
            {
                "id": record.id,
                "text": record.text,
                "score": float(record.score),
                "metadata": dict(record.metadata),
            }
            for record in records
        ]
        known_facts: List[Dict[str, Any]] = []
        if KnowledgeFact is not None:
            for payload in perception_snapshot.knowledge_facts:
                fact = self._normalise_fact_payload(payload)
                if fact is None:
                    continue
                try:
                    known = bool(
                        self.long_term_memory.known_fact(
                            subject=fact.subject,
                            predicate=fact.predicate,
                            obj=fact.obj,
                        )
                    )
                except Exception:  # pragma: no cover - defensive fallback
                    known = False
                if known:
                    known_facts.append(
                        {
                            "subject": fact.subject,
                            "predicate": fact.predicate,
                            "object": fact.obj,
                        }
                    )
        retrieval_context = {
            "query": query_text,
            "records": record_payload,
            "known_facts": known_facts,
        }
        self.last_memory_retrieval = retrieval_context
        if record_payload or known_facts:
            context["memory_retrieval"] = retrieval_context
            context["memory_query"] = query_text
            if record_payload:
                context["memory_records"] = record_payload
            if known_facts:
                context["memory_known_facts"] = known_facts

    def get_long_term_memory_coordinator(self) -> Any | None:
        """Expose the active long-term memory coordinator, if available."""

        return self.long_term_memory

    def get_memory_retrieval(self) -> Dict[str, Any]:
        """Return the last memory retrieval context in a copy-safe form."""

        return {
            "query": self.last_memory_retrieval.get("query"),
            "records": [
                {
                    "id": record.get("id"),
                    "text": record.get("text"),
                    "score": record.get("score"),
                    "metadata": dict(record.get("metadata") or {}),
                }
                for record in self.last_memory_retrieval.get("records", [])
            ],
            "known_facts": [
                dict(fact)
                for fact in self.last_memory_retrieval.get("known_facts", [])
            ],
        }

    def get_retrieved_memory_facts(self) -> List[Dict[str, Any]]:
        """Convenience accessor for the retrieved memory records."""

        return [
            {
                "id": record.get("id"),
                "text": record.get("text"),
                "score": record.get("score"),
                "metadata": dict(record.get("metadata") or {}),
            }
            for record in self.last_memory_retrieval.get("records", [])
        ]

    def get_known_memory_facts(self) -> List[Dict[str, Any]]:
        """Return known facts confirmed against the knowledge repository."""

        return [
            dict(fact)
            for fact in self.last_memory_retrieval.get("known_facts", [])
        ]

    def _determine_knowledge_query(self, stimulus: str, context: Dict[str, Any]) -> str:
        stimulus = (stimulus or "").strip()
        if stimulus:
            return stimulus
        task = context.get("task")
        if isinstance(task, str) and task.strip():
            return task.strip()
        return ""

    @staticmethod
    def _perception_signature(snapshot: PerceptionSnapshot) -> str:
        parts = []
        for name, payload in sorted(snapshot.modalities.items()):
            spikes = payload.get("spike_counts") or []
            parts.append(f"{name}:{','.join(str(int(v)) for v in spikes)}")
        return "|".join(parts)

    def _estimate_novelty(self, snapshot: PerceptionSnapshot) -> float:
        if not snapshot.modalities:
            return 0.0
        novelty = 0.0
        denominator = 0.0
        for name, payload in snapshot.modalities.items():
            spikes = payload.get("spike_counts") or []
            denominator += len(spikes) or 1
            previous_snapshot = self.perception_history[-1] if self.perception_history else self.last_perception
            previous = previous_snapshot.modalities.get(name, {}) if previous_snapshot else {}
            previous_spikes = previous.get("spike_counts") or []
            length = max(len(spikes), len(previous_spikes))
            if length == 0:
                continue
            diff = 0.0
            for idx in range(length):
                current = float(spikes[idx]) if idx < len(spikes) else 0.0
                prior = float(previous_spikes[idx]) if idx < len(previous_spikes) else 0.0
                diff += abs(current - prior)
            novelty += diff / length
        if denominator <= 0:
            denominator = 1.0
        return max(0.0, min(1.0, novelty / denominator))

    def _ensure_motor_backend(self, neurons: int) -> NeuromorphicBackend:
        size = max(1, min(self.max_neurons, int(neurons) if neurons else 1))
        backend = self._motor_backend
        current_size = None
        if backend is not None:
            neuron_obj = getattr(getattr(backend, 'network', None), 'neurons', None)
            current_size = getattr(neuron_obj, 'size', None)
        if backend is None or current_size != size:
            cfg = SpikingNetworkConfig(n_neurons=size, idle_skip=True)
            backend = cfg.create_backend()
            self._motor_backend = backend
        try:
            self._latest_neuromorphic_capability = backend.capability_profile()
        except Exception:
            self._latest_neuromorphic_capability = {}
        backend.reset_state()
        self.motor.spiking_backend = backend
        return backend

    def _channel_to_module(self, channel: str) -> Optional[str]:
        key = channel.lower()
        if key in self._CHANNEL_TO_MODULE:
            return self._CHANNEL_TO_MODULE[key]
        if key in self.topology.module_to_regions:
            return key
        return None

    def _modulator_to_module(self, modulator: str) -> Optional[str]:
        key = modulator.lower()
        if key in self._MODULATOR_TO_MODULE:
            return self._MODULATOR_TO_MODULE[key]
        if key in self.topology.module_to_regions:
            return key
        return None

    def _map_motor_plan_to_regions(
        self,
        weights: Mapping[str, float],
        *,
        modulators: Mapping[str, float] | None = None,
        oscillation_state: Mapping[str, float] | None = None,
    ) -> Dict[str, Dict[str, float]]:
        def _normalise(value: Any) -> float:
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                return 0.0

        module_activity: Dict[str, float] = {}
        for channel, value in weights.items():
            module = self._channel_to_module(channel)
            if not module:
                continue
            module_activity[module] = max(module_activity.get(module, 0.0), _normalise(value))

        if modulators:
            for name, value in modulators.items():
                module = self._modulator_to_module(name)
                if not module:
                    continue
                module_activity[module] = max(module_activity.get(module, 0.0), _normalise(value))

        if oscillation_state:
            numeric = [
                float(val)
                for val in oscillation_state.values()
                if isinstance(val, (int, float))
            ]
            if numeric:
                avg = sum(numeric) / len(numeric)
                module_activity["consciousness"] = max(
                    module_activity.get("consciousness", 0.0), _normalise(avg)
                )

        if not module_activity:
            return {}

        regions = self.topology.project_activity(module_activity)
        connectome_projection = self.connectome.propagate(regions)
        return {
            "modules": module_activity,
            "regions": regions,
            "connectome": connectome_projection,
        }

    def _run_motor_neuromorphic(
        self,
        weights: Dict[str, float],
        intention: str,
        encoding_mode: str,
        modulators: Optional[Dict[str, float]] = None,
        oscillation_state: Optional[Dict[str, float]] = None,
    ) -> Optional[NeuromorphicRunResult]:
        if not self.neuromorphic or (not weights and not modulators):
            return None
        base_channels = ['observe', 'approach', 'withdraw', 'explore']
        extras = [key for key in weights.keys() if key not in base_channels]
        mod_channels: list[str] = []
        if modulators:
            mod_channels = [f"mod_{key}" for key in sorted(modulators.keys())]
        ordering = base_channels + extras + mod_channels
        vector = []
        for key in base_channels + extras:
            value = float(weights.get(key, 0.0))
            vector.append(max(0.0, min(1.0, value)))
        for channel_name in mod_channels:
            original = channel_name[4:]
            value = float(modulators.get(original, 0.0)) if modulators else 0.0
            vector.append(max(0.0, min(1.0, value)))
        backend = self._ensure_motor_backend(len(vector))
        encoder_kwargs: Dict[str, Any] = {}
        decoder_kwargs: Dict[str, Any] = {}
        mode = (encoding_mode or 'rate').lower()
        if mode == 'latency':
            encoder_kwargs['t_scale'] = self.encoding_time_scale
            decoder_kwargs['window'] = float(len(vector) or 1)
        else:
            steps = max(1, self.encoding_steps)
            encoder_kwargs['steps'] = steps
            decoder_kwargs['window'] = float(steps)
            mode = 'rate'
        region_map = self._map_motor_plan_to_regions(
            weights,
            modulators=modulators,
            oscillation_state=oscillation_state,
        )
        metadata = {
            'intention': intention,
            'channels': ordering,
            'weights': dict(weights),
            'modulators': dict(modulators) if modulators else {},
        }
        if oscillation_state:
            metadata['oscillation'] = dict(oscillation_state)
        backend_capability: Dict[str, Any] = {}
        try:
            backend_capability = backend.capability_profile()
            metadata['backend_capability'] = dict(backend_capability)
            self._latest_neuromorphic_capability = dict(backend_capability)
        except Exception:
            backend_capability = {}
        if region_map:
            metadata['module_activity'] = dict(region_map['modules'])
            metadata['regional_activity'] = dict(region_map['regions'])
            metadata['connectome_projection'] = dict(region_map['connectome'])
        result = backend.run_sequence(
            [vector],
            encoding=mode,
            encoder_kwargs=encoder_kwargs,
            decoder='all',
            decoder_kwargs=decoder_kwargs,
            metadata=metadata,
            neuromodulation=oscillation_state,
            reset=True,
        )
        if region_map:
            result.metadata.setdefault('module_activity', dict(region_map['modules']))
            result.metadata.setdefault('regional_activity', dict(region_map['regions']))
            result.metadata.setdefault('connectome_projection', dict(region_map['connectome']))
        self.last_motor_result = result
        return result

    def _compute_oscillation_state(
        self,
        perception: PerceptionSnapshot,
        novelty: float,
    ) -> Dict[str, float]:
        if not self.config.metrics_enabled:
            return {}
        try:
            modalities = max(1, len(perception.modalities) or 1)
            num_osc = max(2, min(4, modalities))
            coupling = 0.4 + 0.6 * max(0.0, min(1.0, novelty))
            stimulus = 1.0 + max(0.0, min(1.0, self.curiosity.drive))
            criticality = 0.8 + 0.2 * max(0.0, min(1.0, self.personality.openness))
            waves = self.oscillations.generate_realistic_oscillations(
                num_oscillators=num_osc,
                duration=max(0.1, 0.1 * modalities),
                sample_rate=200,
                coupling_strength=coupling,
                stimulus=stimulus,
                criticality=criticality,
            )
            if getattr(waves, 'size', 0) == 0:
                return {}
            amplitude = float(np.mean(np.abs(waves)))
            amplitude_norm = float(np.tanh(amplitude))
            if getattr(waves, 'ndim', 0) >= 2 and waves.shape[0] > 1:
                correlation = np.corrcoef(waves)
                mask = ~np.eye(correlation.shape[0], dtype=bool)
                synchrony_index = float(np.mean(np.abs(correlation[mask]))) if mask.any() else 0.0
                modulation = float(np.mean(waves[-1]))
            else:
                synchrony_index = 0.0
                modulation = float(np.mean(waves))
            synchrony_norm = float(np.clip(synchrony_index, 0.0, 1.0))
            spectral = np.fft.rfft(waves, axis=-1)
            spectral_power = np.abs(spectral) ** 2
            mean_power = spectral_power.mean(axis=0)
            freqs = np.fft.rfftfreq(waves.shape[-1], d=1.0 / 200.0)
            if mean_power.size > 0:
                dominant_idx = int(np.argmax(mean_power))
                dominant_frequency = float(freqs[dominant_idx])
                rhythmicity = float(np.tanh(mean_power[dominant_idx]))
            else:
                dominant_frequency = 0.0
                rhythmicity = 0.0
            plasticity_gate = float(np.clip((amplitude_norm + synchrony_norm) * 0.5 + rhythmicity * 0.25, 0.0, 2.0))
            state = {
                'amplitude': amplitude,
                'amplitude_norm': amplitude_norm,
                'synchrony_index': synchrony_index,
                'synchrony_norm': synchrony_norm,
                'modulation': modulation,
                'coupling': coupling,
                'dominant_frequency': dominant_frequency,
                'rhythmicity': rhythmicity,
                'plasticity_gate': plasticity_gate,
            }
            self.last_oscillation_state = state
            return state
        except Exception as exc:
            logger.debug('Oscillation synthesis failed: %s', exc)
            return dict(self.last_oscillation_state)



    def _compose_thought_snapshot(
        self,
        decision: Dict[str, Any],
        memory_refs: List[dict[str, Any]],
    ) -> ThoughtSnapshot:
        plan_steps = list(decision.get("plan", []))
        summary = decision.get("summary") or (
            ', '.join(plan_steps) if plan_steps else decision.get("intention", "")
        )
        return ThoughtSnapshot(
            focus=str(decision.get("focus", decision.get("intention", "unknown"))),
            summary=summary,
            plan=plan_steps,
            confidence=float(decision.get("confidence", 0.5)),
            memory_refs=memory_refs[-3:],
            tags=list(decision.get("tags", [])),
        )

    def _compose_feeling_snapshot(
        self,
        emotion: EmotionSnapshot,
        oscillation_state: Dict[str, float],
        context_features: Dict[str, Any],
    ) -> FeelingSnapshot:
        descriptor = emotion.primary.value.lower()
        valence = float(emotion.dimensions.get("valence", emotion.mood))
        arousal = float(emotion.dimensions.get("arousal", abs(emotion.mood)))
        confidence = max(0.0, min(1.0, 1.0 - float(emotion.decay)))
        context_tags = {
            key
            for key, value in context_features.items()
            if isinstance(value, (int, float)) and value != 0
        }
        for key in oscillation_state:
            context_tags.add(f"osc_{key}")
        return FeelingSnapshot(
            descriptor=descriptor,
            valence=valence,
            arousal=arousal,
            mood=emotion.mood,
            confidence=confidence,
            context_tags=sorted(context_tags),
        )

    def _get_neuromorphic_capability_profile(self) -> Dict[str, Any]:
        if self._latest_neuromorphic_capability:
            return dict(self._latest_neuromorphic_capability)
        backend = getattr(self, "_motor_backend", None)
        if backend is None:
            return {}
        try:
            profile = backend.capability_profile()
        except Exception:
            profile = {}
        self._latest_neuromorphic_capability = dict(profile)
        return dict(profile)

    def _is_gpu_available(self) -> bool:
        if self._cached_gpu_available is not None:
            return bool(self._cached_gpu_available)
        available = False
        try:
            import torch  # type: ignore

            available = bool(torch.cuda.is_available())
        except Exception:
            available = False
        self._cached_gpu_available = available
        return available

    def _hardware_overview(self) -> Dict[str, Any]:
        try:
            snapshot = get_hardware_registry().snapshot()
        except Exception:
            return {}
        overview: Dict[str, Any] = {
            "workers": len(snapshot),
            "accelerators": {},
        }
        for worker_id, caps in snapshot.items():
            hardware = caps.get("hardware") if isinstance(caps, dict) else None
            if isinstance(hardware, dict):
                overview["accelerators"][worker_id] = {
                    "name": hardware.get("name"),
                    "available": hardware.get("available"),
                }
        return overview

    def _determine_pipeline_routing(self) -> Dict[str, Any]:
        plan: Dict[str, Any] = {
            "perception": {"target": "gpu" if self._is_gpu_available() else "cpu"},
            "cognition": {"target": "gpu" if self._is_gpu_available() else "cpu"},
            "symbolic": {"target": "cpu"},
        }
        neuromorphic_capability = self._get_neuromorphic_capability_profile()
        if self.neuromorphic and neuromorphic_capability:
            plan["spiking"] = {
                "target": "neuromorphic" if neuromorphic_capability.get("supports_hardware") else "cpu",
                "capability": neuromorphic_capability,
            }
        else:
            plan["spiking"] = {"target": "cpu"}
        for stage, target in getattr(self.config, "hardware_targets", {}).items():
            plan.setdefault(stage, {})["target"] = target
        plan["environment"] = self._hardware_overview()
        return plan

    def _estimate_resource_signal(self, routing_plan: Dict[str, Any], energy_used: float) -> Dict[str, float]:
        signal: Dict[str, float] = {"cpu": 1.0}
        if routing_plan.get("perception", {}).get("target") == "gpu" or routing_plan.get("cognition", {}).get("target") == "gpu":
            signal["gpu"] = 0.7
        if routing_plan.get("spiking", {}).get("target") == "neuromorphic":
            support = routing_plan["spiking"].get("capability", {}).get("supports_hardware", False)
            signal["neuromorphic"] = 0.9 if support else 0.3
        signal["energy"] = float(max(0.0, energy_used))
        return signal

    @staticmethod
    def _normalise_capability_token(token: Any) -> str:
        if token is None:
            return ""
        text = str(token).strip().lower()
        for prefix in ("focus-", "capability:", "skill:", "strategy:", "intent:", "intent_", "action:"):
            if text.startswith(prefix):
                text = text[len(prefix):]
        sanitized = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")
        return sanitized.strip("_-")

    def _maybe_schedule_exploration_goal(
        self,
        decision: Dict[str, Any],
        cognitive_context: Dict[str, Any],
    ) -> Optional[str]:
        if not self._exploration_goal_queue:
            return None

        intention = str(decision.get("intention") or "").lower()
        confidence = float(decision.get("confidence", 0.0))
        if intention != "observe" or confidence >= 0.35:
            return None
        if cognitive_context.get("task"):
            return None

        state, candidate = self._exploration_goal_queue.popleft()
        try:
            goal_id = self._activate_exploration_goal(state, candidate, decision, cognitive_context)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Failed to schedule exploration goal for %s: %s", state, exc)
            return None
        return goal_id

    def _activate_exploration_goal(
        self,
        state: str,
        candidate: Dict[str, Any],
        decision: Dict[str, Any],
        cognitive_context: Dict[str, Any],
    ) -> str:
        candidate = candidate or {}
        sample = candidate.get("sample") or {}
        raw_goal = sample.get("goal_id") or sample.get("task") or candidate.get("goal") or state
        goal_id = str(raw_goal) if raw_goal else str(state)
        if not goal_id.startswith("explore"):
            goal_id = f"explore:{goal_id}"

        existing_plan = list(decision.get("plan") or [])
        decision["plan"] = [goal_id] + [step for step in existing_plan if step != goal_id]
        decision["intention"] = "explore"
        decision["goal"] = goal_id
        decision["confidence"] = max(float(decision.get("confidence", 0.0)), 0.4)
        weights = decision.setdefault("weights", {})
        weights["explore"] = max(float(weights.get("explore", 0.0)), decision["confidence"])
        decision.setdefault("tags", [])
        if "explore" not in decision["tags"]:
            decision["tags"].append("explore")
        decision.setdefault("thought_trace", []).append("self-directed-exploration")

        policy_metadata = decision.setdefault("policy_metadata", {})
        policy_metadata.setdefault("policy", policy_metadata.get("policy", "manual-override"))
        policy_metadata["exploration_goal"] = {"id": goal_id, "state": state}
        if candidate.get("metadata"):
            policy_metadata["exploration_metadata"] = dict(candidate["metadata"])
        if sample:
            policy_metadata["exploration_sample"] = dict(sample)
        policy_metadata["exploration_trigger"] = "idle-cycle"

        cognitive_context["task"] = goal_id
        cognitive_context["exploration_origin_state"] = state

        self._scheduled_exploration_goals[goal_id] = state
        return goal_id

    def process_cycle(self, input_data: Dict[str, Any]) -> BrainCycleResult:
        """Run a single perception-cognition-action cycle."""

        if self._metrics_collector is not None and self.config.enable_self_evolution:
            try:
                self._metrics_collector.start("whole_brain_cycle")
            except Exception:
                pass

        self.cycle_index += 1
        input_data = dict(input_data or {})
        use_neuromorphic = self.config.use_neuromorphic
        if use_neuromorphic != self.neuromorphic:
            use_neuromorphic = self.neuromorphic
            self.config.use_neuromorphic = self.neuromorphic

        if input_data.get("reset_streams"):
            self._stream_state.clear()
        drop_streams = input_data.get("drop_streams")
        if isinstance(drop_streams, Iterable) and not isinstance(drop_streams, (str, bytes)):
            for key in list(drop_streams):
                self._stream_state.pop(key, None)

        perception: Dict[str, Dict[str, Any]] = {}
        energy_used = 0.0
        idle_skipped = 0
        cycle_errors: List[str] = []
        cycle_telemetry: Dict[str, Any] = {
            "cycle_index": self.cycle_index,
            "use_neuromorphic": bool(use_neuromorphic),
        }
        routing_plan = self._determine_pipeline_routing()
        cycle_telemetry["routing_plan"] = routing_plan
        self.last_memory_retrieval = {}

        def _resolve_signal(*keys: str) -> Any:
            for key in keys:
                if key in input_data:
                    return input_data[key]
            return None

        def _flatten_signal(value: Any) -> list[float]:
            if isinstance(value, Real):
                return [float(value)]
            if hasattr(value, "tolist"):
                return _flatten_signal(value.tolist())
            if isinstance(value, (list, tuple)):
                flat: list[float] = []
                for item in value:
                    flat.extend(_flatten_signal(item))
                return flat
            logger.debug("Unsupported sensory signal type: %s", type(value).__name__)
            return []

        def _consume_stream(modality: str, source: Any) -> Any:
            if source is None:
                return None
            try:
                if callable(source):
                    return source()
                if hasattr(source, "__next__"):
                    return next(source)
                if isinstance(source, (list, tuple, deque)):
                    return source[-1] if source else None
                if isinstance(source, dict):
                    for key in ("latest", "value", "data"):
                        if key in source:
                            return source[key]
                if isinstance(source, Iterable) and not isinstance(source, (str, bytes)):
                    iterator = iter(source)
                    return next(iterator)
            except StopIteration:
                return None
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("Stream consumption failed for %s: %s", modality, exc)
                cycle_errors.append(f"stream:{modality}:{exc}")
                return None
            return source

        sensory_inputs: Dict[str, Any] = {}

        def _register_input(modality: str, value: Any, source: str) -> None:
            if value is None:
                return
            sensory_inputs[modality] = value
            modality_sources = cycle_telemetry.setdefault("modalities", {})
            modality_sources[modality] = source
            if input_data.get("persist_streams", True):
                self._stream_state[modality] = value

        streams = input_data.get("streams")
        if isinstance(streams, dict):
            for modality, source in streams.items():
                value = _consume_stream(modality, source)
                if value is not None:
                    _register_input(modality, value, "stream")

        stream_events = input_data.get("stream_events")
        if isinstance(stream_events, Iterable) and not isinstance(stream_events, (str, bytes)):
            for event in stream_events:
                try:
                    if not isinstance(event, dict):
                        continue
                    modality = event.get("modality") or event.get("name")
                    if not modality:
                        continue
                    value = event.get("value", event.get("data"))
                    if value is None:
                        continue
                    _register_input(modality, value, "event")
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.debug("Failed to process stream event %s: %s", event, exc)
                    cycle_errors.append(f"stream-event:{exc}")

        vision_signal = _resolve_signal("vision", "image")
        if vision_signal is not None:
            _register_input("vision", vision_signal, "direct")
        auditory_signal = _resolve_signal("auditory", "sound", "audio")
        if auditory_signal is not None:
            _register_input("auditory", auditory_signal, "direct")
        somatosensory_signal = _resolve_signal("somatosensory", "touch")
        if somatosensory_signal is not None:
            _register_input("somatosensory", somatosensory_signal, "direct")

        if input_data.get("use_cached_streams", True):
            for modality, cached_value in self._stream_state.items():
                sensory_inputs.setdefault(modality, cached_value)
                modality_sources = cycle_telemetry.setdefault("modalities", {})
                modality_sources.setdefault(modality, "cached")

        if use_neuromorphic:

            def _compress(vector: list[float], target: int | None = None) -> list[float]:
                if not vector:
                    return vector
                limit = self.max_neurons
                if target is not None:
                    limit = max(1, min(limit, int(target)))
                if len(vector) <= limit:
                    return vector
                chunk = max(1, math.ceil(len(vector) / limit))
                compressed: list[float] = []
                for i in range(0, len(vector), chunk):
                    segment = vector[i : i + chunk]
                    if segment:
                        compressed.append(sum(segment) / len(segment))
                    if len(compressed) == limit:
                        break
                return compressed or vector[:limit]

            def _select_bucket(length: int) -> int:
                if length <= 0:
                    return 0
                if length >= self.max_neurons:
                    return self.max_neurons
                power = 1 << (length - 1).bit_length()
                return min(self.max_neurons, power)

            def _encode_and_run(signal: Any, modality: str) -> None:
                nonlocal energy_used, idle_skipped
                try:
                    encoded = self.perception_pipeline.encode(modality, signal)
                except Exception as exc:
                    logger.debug("Perception encoding failed for %s: %s", modality, exc)
                    cycle_errors.append(f"encode:{modality}:{exc}")
                    encoded = EncodedSignal()
                source_vector = encoded.vector or _flatten_signal(signal)
                vector = _compress(source_vector)
                if not vector:
                    return
                base_signal = _flatten_signal(signal)
                bucket = _select_bucket(len(base_signal) or len(vector))
                if len(vector) > bucket:
                    vector = _compress(vector, target=bucket)
                if bucket == 0:
                    return
                if len(vector) < bucket:
                    vector = vector + [0.0] * (bucket - len(vector))

                encoding_mode = (self.neuromorphic_encoding or "rate").lower()
                cache_key = (bucket, encoding_mode)
                backend = self._spiking_cache.get(cache_key)
                if backend is None:
                    cfg = SpikingNetworkConfig(n_neurons=bucket, idle_skip=True)
                    backend = cfg.create_backend()
                    self._spiking_cache[cache_key] = backend
                else:
                    backend.reset_state()
                self._spiking_cache.move_to_end(cache_key)
                if len(self._spiking_cache) > self.max_cache_size:
                    self._spiking_cache.popitem(last=False)

                encoder_kwargs: Dict[str, Any] = {}
                decoder_kwargs: Dict[str, Any] = {}
                if encoding_mode == "latency":
                    encoder_kwargs["t_scale"] = self.encoding_time_scale
                    decoder_kwargs["window"] = float(len(vector) or 1)
                elif encoding_mode == "rate":
                    steps = max(1, self.encoding_steps)
                    encoder_kwargs["steps"] = steps
                    decoder_kwargs["window"] = float(steps)

                try:
                    result = backend.run_sequence(
                        [vector],
                        encoding=encoding_mode if encoding_mode in {"rate", "latency"} else None,
                        encoder_kwargs=encoder_kwargs,
                        decoder="all",
                        decoder_kwargs=decoder_kwargs,
                        metadata={"modality": modality},
                        reset=False,
                    )
                except Exception as exc:  # pragma: no cover - backend failure handling
                    logger.debug("Neuromorphic backend failed for %s: %s", modality, exc)
                    cycle_errors.append(f"spiking:{modality}:{exc}")
                    return

                raw_counts = list(result.spike_counts)
                entry: Dict[str, Any] = {
                    "spike_counts": raw_counts,
                    "spike_events": result.spike_events,
                    "average_rate": result.average_rate,
                    "vector": vector,
                    "metadata": {
                        "energy_used": result.energy_used,
                        "idle_skipped": result.idle_skipped,
                        "encoding": encoding_mode,
                    },
                }
                if modality in {"auditory", "audio", "sound"}:
                    raw_waveform = _flatten_signal(signal)
                    if raw_waveform:
                        entry["raw_waveform"] = raw_waveform
                    if isinstance(signal, dict):
                        meta = entry.setdefault("metadata", {})
                        for key in ("audio_path", "file_path", "path"):
                            if key in signal and key not in meta:
                                meta[key] = signal[key]
                        if "transcript" in signal and "transcript" not in meta:
                            meta["transcript"] = signal["transcript"]
                if encoded.features:
                    entry["features"] = encoded.features
                if encoded.metadata:
                    entry["metadata"].update(encoded.metadata)
                base_for_counts = list(base_signal) if base_signal else []
                if len(base_for_counts) < len(raw_counts):
                    base_for_counts.extend([0.0] * (len(raw_counts) - len(base_for_counts)))
                if base_for_counts and raw_counts:
                    max_value = max(base_for_counts)
                    min_value = min(base_for_counts)
                    if abs(max_value - min_value) <= 1e-9:
                        peak_index = base_for_counts.index(max_value)
                        normalised = [
                            1 if idx == peak_index else 0 for idx in range(len(raw_counts))
                        ]
                    else:
                        normalised = [
                            1 if abs(value - max_value) <= 1e-9 else 0
                            for value in base_for_counts[: len(raw_counts)]
                        ]
                    entry["metadata"]["raw_spike_counts"] = raw_counts
                    entry["spike_counts"] = normalised
                perception[modality] = entry
                energy_used += float(result.energy_used)
                idle_skipped += int(result.idle_skipped)

            for modality, signal in sensory_inputs.items():
                _encode_and_run(signal, modality)
        else:
            for modality, signal in sensory_inputs.items():
                try:
                    encoded = self.perception_pipeline.encode(modality, signal)
                except Exception as exc:
                    logger.debug("Perception encoding failed for %s: %s", modality, exc)
                    cycle_errors.append(f"encode:{modality}:{exc}")
                    encoded = EncodedSignal()
                vector = encoded.vector or _flatten_signal(signal)
                entry: Dict[str, Any] = {
                    "vector": vector,
                    "metadata": {"encoding": "analytic"},
                }
                if modality in {"auditory", "audio", "sound"}:
                    raw_waveform = _flatten_signal(signal)
                    if raw_waveform:
                        entry["raw_waveform"] = raw_waveform
                    if isinstance(signal, dict):
                        meta = entry.setdefault("metadata", {})
                        for key in ("audio_path", "file_path", "path"):
                            if key in signal and key not in meta:
                                meta[key] = signal[key]
                        if "transcript" in signal and "transcript" not in meta:
                            meta["transcript"] = signal["transcript"]
                if encoded.features:
                    entry["features"] = encoded.features
                if encoded.metadata:
                    entry["metadata"].update(encoded.metadata)
                perception[modality] = entry

        if "auditory" in perception and "audio" not in perception:
            perception["audio"] = dict(perception["auditory"])
        if "somatosensory" in perception and "touch" not in perception:
            perception["touch"] = dict(perception["somatosensory"])

        perception_snapshot = PerceptionSnapshot(modalities=dict(perception))
        if getattr(self, "semantic_bridge", None) is not None:
            try:
                agent_identifier = str(input_data.get("agent_id", "whole_brain"))
                semantic_output = self.semantic_bridge.process(
                    perception_snapshot,
                    agent_id=agent_identifier,
                    cycle_index=self.cycle_index,
                )
            except Exception:  # pragma: no cover - semantic decoding is optional
                logger.debug("Semantic bridge processing failed.", exc_info=True)
            else:
                if semantic_output.semantic_annotations:
                    perception_snapshot.semantic.update(semantic_output.semantic_annotations)
                if semantic_output.fused_embedding:
                    perception_snapshot.fused_embedding = list(semantic_output.fused_embedding)
                if semantic_output.modality_embeddings:
                    perception_snapshot.modality_embeddings.update(semantic_output.modality_embeddings)
                if semantic_output.knowledge_facts:
                    perception_snapshot.knowledge_facts.extend(semantic_output.knowledge_facts)
                    self._record_long_term_facts(
                        semantic_output.knowledge_facts,
                        source="semantic_bridge",
                        metadata={
                            "agent_id": agent_identifier,
                            "cycle_index": self.cycle_index,
                        },
                    )
        if self.knowledge_base is not None and perception_snapshot.knowledge_facts:
            facts_to_ingest: List[Any] = []
            if KnowledgeFact is not None:
                for payload in perception_snapshot.knowledge_facts:
                    fact = self._normalise_fact_payload(payload)
                    if fact is not None:
                        facts_to_ingest.append(fact)
            if facts_to_ingest:
                try:
                    self.knowledge_base.ingest_facts(facts_to_ingest)
                except Exception:
                    logger.debug("Knowledge base ingestion failed.", exc_info=True)
        novelty = self._estimate_novelty(perception_snapshot)
        if self.config.enable_curiosity_feedback:
            self.curiosity.update(novelty, self.personality)
        else:
            self.curiosity.decay()
            self.curiosity.last_novelty = novelty

        oscillation_state = self._compute_oscillation_state(perception_snapshot, novelty)

        raw_context = input_data.get("context", {})
        cognitive_context: Dict[str, Any] = dict(raw_context) if isinstance(raw_context, dict) else {}
        if "task" in input_data and "task" not in cognitive_context:
            cognitive_context["task"] = input_data["task"]
        if "is_salient" in input_data:
            cognitive_context["salience"] = bool(input_data.get("is_salient"))
        cognitive_context.setdefault("novelty", novelty)
        cognitive_context.setdefault("cycle_index", self.cycle_index)
        cognitive_context.setdefault("energy_used", float(energy_used))

        context_features: Dict[str, float] = {}
        for key, value in cognitive_context.items():
            if isinstance(value, Real):
                context_features[key] = float(value)
        context_features.setdefault("novelty", novelty)
        capability_overview = self.self_model.capability_summary()
        if capability_overview:
            top_capabilities = sorted(
                capability_overview.items(),
                key=lambda item: item[1].get("weight", 0.5),
                reverse=True,
            )
            cognitive_context["self_capabilities"] = [
                {
                    "name": name,
                    "weight": float(data.get("weight", 0.5)),
                    "success_rate": float(data.get("success_rate", 0.0)),
                }
                for name, data in top_capabilities[:4]
            ]
            if top_capabilities:
                primary_weight = float(top_capabilities[0][1].get("weight", 0.5))
                context_features.setdefault("capability_bias", primary_weight)
                context_features.setdefault("intention_capability_hint", primary_weight)
            for name, data in top_capabilities[:3]:
                context_features[f"cap_{self._normalise_capability_token(name)}"] = float(
                    data.get("weight", 0.5)
                )

        text_signal = _resolve_signal("text", "language", "stimulus", "narrative")
        text_stimulus = str(text_signal) if text_signal is not None else ""
        knowledge_query: str = ""
        knowledge_context: Dict[str, Any] | None = None
        if self.knowledge_base is not None:
            cognitive_context.setdefault("knowledge_base", self.knowledge_base)
            knowledge_query = self._determine_knowledge_query(text_stimulus, cognitive_context)
            if knowledge_query:
                try:
                    knowledge_context = self.knowledge_base.query(
                        knowledge_query,
                        semantic=True,
                        top_k=5,
                    )
                except Exception:
                    knowledge_context = {}
                if knowledge_context:
                    try:
                        context_payload = dict(knowledge_context)
                    except Exception:
                        context_payload = {"primary": knowledge_context}
                    self._inject_causal_context(
                        context_payload,
                        cognitive_context,
                        knowledge_query=knowledge_query,
                    )
                    cognitive_context["knowledge_context"] = context_payload
                    cognitive_context.setdefault("knowledge_query", knowledge_query)
                elif self.config.enable_self_learning:
                    try:
                        self.self_learning.register_knowledge_gap(
                            knowledge_query,
                            context={
                                "source": "knowledge-base",
                                "cycle_index": self.cycle_index,
                            },
                            reason="knowledge-base-empty",
                            priority=1.2,
                        )
                    except Exception:  # pragma: no cover - curiosity optional in tests
                        logger.debug(
                            "Failed to register knowledge gap for query %s", knowledge_query,
                            exc_info=True,
                        )
        perception_summary_hint = self._summarise_perception(perception_snapshot)
        raw_context_dict = raw_context if isinstance(raw_context, dict) else {}
        intent_hint = raw_context_dict.get("intent") if isinstance(raw_context_dict, dict) else None
        if not intent_hint and isinstance(raw_context_dict, dict):
            intent_hint = raw_context_dict.get("intention")
        if not intent_hint:
            intent_hint = input_data.get("intent") or input_data.get("intention")
        if not intent_hint:
            intent_hint = cognitive_context.get("task")
        self._augment_context_with_long_term_memory(
            cognitive_context,
            perception_summary_hint,
            perception_snapshot,
            intent_hint=intent_hint,
            fallback_text=text_stimulus,
        )
        if (
            self.config.enable_self_learning
            and knowledge_query
            and not (knowledge_context or {})
        ):
            retrieval_snapshot = dict(self.last_memory_retrieval or {})
            records = retrieval_snapshot.get("records") or []
            known_facts = retrieval_snapshot.get("known_facts") or []
            if not records and not known_facts:
                try:
                    self.self_learning.register_knowledge_gap(
                        knowledge_query,
                        context={
                            "source": "long-term-memory",
                            "cycle_index": self.cycle_index,
                        },
                        reason="memory-retrieval-empty",
                        priority=1.1,
                    )
                except Exception:  # pragma: no cover - optional subsystem
                    logger.debug(
                        "Failed to register long-term memory gap for %s", knowledge_query,
                        exc_info=True,
                    )
        if perception_summary_hint and "perception_summary_hint" not in cognitive_context:
            cognitive_context["perception_summary_hint"] = perception_summary_hint
        emotional_state = self.emotion.react(text_stimulus, context_features, self.config)
        emotion_snapshot = EmotionSnapshot(
            primary=emotional_state.emotion,
            intensity=float(emotional_state.intensity),
            mood=float(self.emotion.mood),
            dimensions=dict(emotional_state.dimensions),
            context=dict(emotional_state.context_weights),
            decay=float(emotional_state.decay),
            intent_bias=dict(emotional_state.intent_bias),
        )

        personality_snapshot = PersonalityProfile(
            openness=float(self.personality.openness),
            conscientiousness=float(self.personality.conscientiousness),
            extraversion=float(self.personality.extraversion),
            agreeableness=float(self.personality.agreeableness),
            neuroticism=float(self.personality.neuroticism),
        )

        learning_prediction: Dict[str, float] = {}
        reward_signal: float | None = None
        if self.config.enable_self_learning:
            signature = self._perception_signature(perception_snapshot) or f"cycle-{self.cycle_index}"
            usage = {
                "cpu": float(energy_used),
                "memory": float(
                    sum(
                        len(payload.get("spike_counts") or [])
                        for payload in perception_snapshot.modalities.values()
                    )
                ),
            }
            reward = emotional_state.dimensions.get("valence", 0.0) * emotional_state.intensity
            reward += context_features.get("safety", 0.0) * 0.1
            reward -= context_features.get("threat", 0.0) * 0.2
            sample = {
                "state": signature,
                "agent_id": str(input_data.get("agent_id", "whole_brain")),
                "usage": usage,
                "reward": max(-1.0, min(1.0, reward)),
            }
            reward_signal = sample["reward"]
            learning_prediction = self.self_learning.curiosity_driven_learning(sample) or {}
        else:
            signature = f"cycle-{self.cycle_index}"

        if self.config.enable_self_learning:
            pending_candidates = self.self_learning.consume_exploration_candidates()
            if pending_candidates:
                for state, candidate in pending_candidates.items():
                    metadata = candidate.get("metadata", {}) if isinstance(candidate, dict) else {}
                    priority = float(metadata.get("priority", 1.0)) if metadata else 1.0
                    if not self._exploration_goal_queue:
                        self._exploration_goal_queue.append((state, candidate))
                    else:
                        inserted = False
                        for idx, (_, existing) in enumerate(self._exploration_goal_queue):
                            existing_meta = (
                                existing.get("metadata", {}) if isinstance(existing, dict) else {}
                            )
                            existing_priority = float(existing_meta.get("priority", 1.0))
                            if priority > existing_priority:
                                self._exploration_goal_queue.insert(idx, (state, candidate))
                                inserted = True
                                break
                        if not inserted:
                            self._exploration_goal_queue.append((state, candidate))
            cycle_telemetry["exploration_queue"] = len(self._exploration_goal_queue)

        meta_skill_suggestions: List[Dict[str, Any]] = []
        if self.config.enable_self_learning:
            self.meta_learning.bind_policy(self.cognition.policy)
            meta_skill_suggestions = self.meta_learning.inject_suggestions(
                cognitive_context,
                perception_snapshot,
                history=list(self.decision_history),
            )
            if meta_skill_suggestions:
                cycle_telemetry["meta_skill_suggestions"] = list(meta_skill_suggestions)

        policy_override: Optional[CognitivePolicy] = None
        if self.neuromorphic and (
            input_data.get("streams") or input_data.get("stream_events")
        ):
            if not isinstance(self.cognition.policy, HeuristicCognitivePolicy):
                policy_override = self.cognition.policy
                self.cognition.set_policy(HeuristicCognitivePolicy())

        decision = self.cognition.decide(
            perception_snapshot,
            emotion_snapshot,
            self.personality,
            self.curiosity,
            learning_prediction if learning_prediction else None,
            cognitive_context,
        )
        if policy_override is not None:
            self.cognition.set_policy(policy_override)
        scheduled_exploration = self._maybe_schedule_exploration_goal(
            decision, cognitive_context
        )
        if scheduled_exploration:
            cycle_telemetry["scheduled_exploration_goal"] = scheduled_exploration
        if meta_skill_suggestions:
            decision.setdefault("policy_metadata", {}).setdefault(
                "meta_skill_suggestions", list(meta_skill_suggestions)
            )
        capability_signals: Dict[str, float] = {}
        capability_display_signals: Dict[str, float] = {}
        intention_label = str(decision.get("intention") or "").strip()
        intention_key = self._normalise_capability_token(decision.get("intention"))
        base_confidence = float(decision.get("confidence", 0.5))
        if intention_label:
            capability_display_signals[intention_label] = base_confidence
        if intention_key:
            capability_signals[intention_key] = base_confidence
        for tag in decision.get("tags", []):
            tag_norm = self._normalise_capability_token(tag)
            tag_label = str(tag).strip()
            if tag_label:
                capability_display_signals.setdefault(tag_label, base_confidence)
            if not tag_norm:
                continue
            capability_signals[tag_norm] = max(capability_signals.get(tag_norm, 0.0), base_confidence)
        for step in decision.get("plan", []):
            step_norm = self._normalise_capability_token(step)
            step_label = str(step).strip()
            if step_label and step_label not in capability_display_signals:
                capability_display_signals[step_label] = min(0.4, base_confidence)
            if step_norm and step_norm not in capability_signals:
                capability_signals[step_norm] = min(0.4, base_confidence)
        capability_weight = self.self_model.capability_weight(intention_key) if intention_key else 0.5
        decision["confidence"] = max(0.0, min(1.0, base_confidence * (0.6 + 0.7 * capability_weight)))
        decision["capability_bias"] = capability_weight
        if capability_display_signals:
            decision["capability_signals"] = dict(capability_display_signals)
        decision.setdefault("policy_metadata", {}).setdefault("capability_weight", capability_weight)
        decision["routing_plan"] = routing_plan
        cognitive_context["selected_capability"] = {
            "name": decision.get("intention"),
            "weight": capability_weight,
        }
        context_features["intention_capability"] = capability_weight
        intention = decision["intention"]
        if oscillation_state:
            decision["oscillation_state"] = dict(oscillation_state)

        def _clamp(value: Any) -> float:
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                return 0.0

        modulators: Dict[str, float] = {
            "novelty": _clamp(novelty),
            "curiosity": _clamp(self.curiosity.drive),
            "fatigue": _clamp(self.curiosity.fatigue),
            "confidence": _clamp(decision.get("confidence")),
            "valence": _clamp((emotion_snapshot.dimensions.get("valence", 0.0) + 1.0) / 2.0),
            "arousal": _clamp(emotion_snapshot.dimensions.get("arousal", 0.0)),
            "mood": _clamp((emotion_snapshot.mood + 1.0) / 2.0),
        }
        modulators["capability_bias"] = _clamp(capability_weight)
        if "safety" in context_features:
            modulators["safety"] = _clamp(context_features["safety"])
        if "threat" in context_features:
            modulators["threat"] = _clamp(context_features["threat"])
        if self.config.enable_personality_modulation:
            modulators["openness"] = _clamp(self.personality.openness)
            modulators["conscientiousness"] = _clamp(self.personality.conscientiousness)
        if oscillation_state:
            for key, value in oscillation_state.items():
                if isinstance(value, (int, float)):
                    modulators[f"osc_{key}"] = _clamp(value)
        if capability_signals:
            for cap_name in list(capability_signals.keys())[:4]:
                modulators[f"cap_{cap_name}"] = _clamp(self.self_model.capability_weight(cap_name))

        plan_parameters: Dict[str, Any] = {}
        if modulators:
            plan_parameters["modulators"] = dict(modulators)
        if oscillation_state:
            plan_parameters["oscillation"] = dict(oscillation_state)
        if capability_signals:
            plan_parameters["capability_signals"] = {
                name: float(value) for name, value in capability_signals.items()
            }
            if capability_display_signals:
                plan_parameters["capability_labels"] = dict(capability_display_signals)
            plan_parameters["capability_bias"] = capability_weight

        motor_result: Optional[NeuromorphicRunResult] = None
        if decision.get("weights"):
            motor_result = self._run_motor_neuromorphic(
                decision["weights"],
                intention,
                self.neuromorphic_encoding or "rate",
                modulators,
                oscillation_state,
            )
            if motor_result:
                decision["motor_channels"] = list(motor_result.metadata.get("channels", []))
                decision["motor_spike_counts"] = list(motor_result.spike_counts)
        elif self.last_motor_result is not None:
            self.last_motor_result = None

        if decision.get("weights"):
            plan_parameters["weights"] = {
                key: float(value) for key, value in decision["weights"].items()
            }
        balance_hint = self.cerebellum.balance_control(f"novelty:{novelty:.3f}")
        plan_parameters["cerebellum_hint"] = balance_hint
        if motor_result:
            plan_parameters["neuromorphic_result"] = motor_result
            plan_parameters["motor_channels"] = list(motor_result.metadata.get("channels", []))
            if motor_result.average_rate:
                plan_parameters["motor_rate"] = list(motor_result.average_rate)

        try:
            plan = self.motor.plan_movement(intention, parameters=plan_parameters)
        except Exception as exc:  # pragma: no cover - defensive planning fallback
            logger.debug("Motor planning failed: %s", exc)
            cycle_errors.append(f"motor-plan:{exc}")
            plan = MotorPlan(
                intention=intention,
                stages=[f"fallback_{intention}"],
                parameters=dict(plan_parameters),
                metadata={
                    "plan_summary": f"fallback_{intention}",
                    "fallback": True,
                    "error": str(exc),
                },
            )
        if motor_result:
            plan.metadata["neuromorphic"] = motor_result.to_dict()
        try:
            action = self.motor.execute_action(plan)
        except Exception as exc:  # pragma: no cover - defensive execution fallback
            logger.debug("Motor execution failed: %s", exc)
            cycle_errors.append(f"motor-execute:{exc}")
            action = MotorExecutionResult(False, str(exc), telemetry={}, error=str(exc))

        external_feedback = None
        for key in ("motor_feedback", "execution_feedback", "actuator_feedback", "sensor_feedback"):
            if key in input_data:
                external_feedback = input_data[key]
                break
        if external_feedback is not None:
            try:
                self.motor.train(external_feedback)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("Motor training from external feedback failed: %s", exc)

        execution_feedback = action if isinstance(action, MotorExecutionResult) else None
        execution_metrics = self.motor.parse_feedback_metrics(execution_feedback, base_reward=reward_signal)
        external_metrics = (
            self.motor.parse_feedback_metrics(external_feedback, base_reward=reward_signal)
            if external_feedback is not None
            else {}
        )
        feedback_metrics: Dict[str, float] = dict(execution_metrics)
        for key, value in external_metrics.items():
            if key in feedback_metrics and feedback_metrics[key] not in {0.0}:
                feedback_metrics[key] = (feedback_metrics[key] + value) / 2.0
            else:
                feedback_metrics[key] = value
        if reward_signal is not None and "reward" not in feedback_metrics:
            feedback_metrics["reward"] = float(reward_signal)
        if feedback_metrics:
            plan_parameters.setdefault("feedback_metrics", dict(feedback_metrics))
            plan.metadata["feedback_metrics"] = dict(feedback_metrics)
            decision["feedback_metrics"] = dict(feedback_metrics)

        exploration_goal_id: Optional[str] = None
        exploration_state: Optional[str] = None
        if isinstance(decision.get("goal"), str):
            exploration_goal_id = str(decision.get("goal"))
        policy_meta = decision.get("policy_metadata")
        if isinstance(policy_meta, Mapping):
            goal_meta = policy_meta.get("exploration_goal")
            if isinstance(goal_meta, Mapping):
                exploration_goal_id = str(goal_meta.get("id") or exploration_goal_id or "") or None
                exploration_state = goal_meta.get("state") or exploration_state
        if exploration_goal_id:
            exploration_state = exploration_state or self._scheduled_exploration_goals.get(exploration_goal_id)
        if exploration_goal_id and exploration_state:
            metrics_payload: Optional[Mapping[str, Any]] = (
                dict(feedback_metrics) if feedback_metrics else None
            )
            # Exploration goals can span multiple cycles; only consume them once we
            # receive an explicit terminal signal from the environment/controller.
            done_signal = None
            if metrics_payload is not None:
                done_signal = metrics_payload.get("done", metrics_payload.get("extra_done"))
            done_reached = False
            if isinstance(done_signal, Real):
                done_reached = float(done_signal) >= 0.5

            if done_reached:
                outcome_success: Optional[bool] = None
                if metrics_payload:
                    if "success" in metrics_payload:
                        try:
                            outcome_success = bool(metrics_payload["success"])
                        except Exception:  # pragma: no cover - defensive conversion
                            outcome_success = None
                    elif "reward" in metrics_payload:
                        try:
                            outcome_success = float(metrics_payload["reward"]) > 0
                        except Exception:  # pragma: no cover - defensive conversion
                            outcome_success = None
                    elif "success_rate" in metrics_payload:
                        try:
                            outcome_success = float(metrics_payload["success_rate"]) >= 0.5
                        except Exception:  # pragma: no cover - defensive conversion
                            outcome_success = None
                if outcome_success is None and isinstance(action, MotorExecutionResult):
                    outcome_success = bool(action.success)
                if outcome_success is None:
                    outcome_success = False
                fallback_metrics: Optional[Mapping[str, Any]] = metrics_payload
                if fallback_metrics is None and reward_signal is not None:
                    fallback_metrics = {"reward": float(reward_signal)}
                self.self_learning.record_exploration_outcome(
                    exploration_state,
                    bool(outcome_success),
                    fallback_metrics,
                )
                self._scheduled_exploration_goals.pop(exploration_goal_id, None)

        meta_update: Optional[Dict[str, Any]] = None
        if self.config.enable_self_learning:
            meta_update = self.meta_learning.record_outcome(
                cycle_index=self.cycle_index,
                state_signature=signature,
                decision=decision,
                feedback_metrics=feedback_metrics,
                reward_signal=reward_signal,
                cognitive_context=cognitive_context,
            )
            if meta_update:
                cycle_telemetry["meta_learning_update"] = dict(meta_update)

        curiosity_snapshot = CuriosityState(
            drive=self.curiosity.drive,
            novelty_preference=self.curiosity.novelty_preference,
            fatigue=self.curiosity.fatigue,
            last_novelty=self.curiosity.last_novelty,
        )

        intent = CognitiveIntent(
            intention=intention,
            salience=bool(input_data.get("is_salient", False)),
            plan=list(decision.get("plan", [])),
            confidence=float(decision.get("confidence", 0.0)),
            weights=dict(decision.get("weights", {})),
            tags=list(decision.get("tags", [])),
        )

        thought_snapshot = self._compose_thought_snapshot(
            decision,
            self.cognition.recall(),
        )
        feeling_snapshot = self._compose_feeling_snapshot(
            emotion_snapshot,
            oscillation_state,
            context_features,
        )

        perception_summary = decision.get("perception_summary") or self._summarise_perception(
            perception_snapshot
        )
        module_activity = self._compose_module_activity(
            perception_summary,
            emotion_snapshot,
            self.curiosity,
            decision,
            oscillation_state,
            motor_result,
        )
        topology_snapshot = self.topology.build_snapshot(module_activity, self.connectome)
        self.last_topology = topology_snapshot
        spiking_plan = routing_plan.get("spiking")
        if isinstance(spiking_plan, dict):
            connectome_keys = list(topology_snapshot.get("connectome", {}).keys())
            if connectome_keys:
                spiking_plan.setdefault("connectome_partitions", connectome_keys[: min(8, len(connectome_keys))])

        metrics: Dict[str, float] = {}
        if self.config.metrics_enabled:
            metrics.update({"modalities": float(len(perception_snapshot.modalities))})
            metrics.update(
                {
                    "novelty_signal": novelty,
                    "energy_used": float(energy_used),
                    "idle_skipped": float(idle_skipped),
                    "cycle_index": float(self.cycle_index),
                }
            )
            metrics.update(self.curiosity.as_metrics())
            metrics.update(emotion_snapshot.as_metrics())
            if oscillation_state:
                metrics.update({f"osc_{k}": float(v) for k, v in oscillation_state.items()})
            if motor_result:
                metrics["motor_energy"] = float(motor_result.energy_used)
                metrics["motor_idle_skipped"] = float(motor_result.idle_skipped)
                if motor_result.spike_counts:
                    metrics["motor_spike_avg"] = float(
                        sum(motor_result.spike_counts) / len(motor_result.spike_counts)
                    )
                if motor_result.average_rate:
                    metrics["motor_rate_avg"] = float(
                        sum(motor_result.average_rate) / len(motor_result.average_rate)
                    )
            if feedback_metrics:
                metrics.update({f"feedback_{k}": float(v) for k, v in feedback_metrics.items()})
            intent_metrics = intent.as_metrics()
            metrics["intent_confidence"] = intent_metrics.get(
                "intent_confidence", intent.confidence
            )
            weights = decision.get("weights", {})
            for key in ("approach", "withdraw", "explore", "observe"):
                metrics[f"strategy_bias_{key}"] = float(weights.get(key, 0.0))
            if self.config.enable_plan_logging:
                metrics.update({k: v for k, v in intent_metrics.items() if k != "intent_confidence"})
                metrics["plan_length"] = float(len(decision.get("plan", [])))
            if learning_prediction:
                metrics.update(
                    {f"learning_{k}": float(v) for k, v in learning_prediction.items()}
                )
            if cycle_errors:
                metrics["cycle_error_count"] = float(len(cycle_errors))
            metrics.update(
                {f"layer_{name}": float(value) for name, value in topology_snapshot["layers"].items()}
            )

        policy_metadata = dict(decision.get("policy_metadata", {}))
        if policy_metadata.get("policy") == "reinforcement":
            policy_metadata["mode"] = "reinforcement"
            policy_metadata["policy"] = "production"
            decision["policy_metadata"] = dict(policy_metadata)
        resource_signal = self._estimate_resource_signal(routing_plan, energy_used)
        cycle_telemetry["resource_signal"] = dict(resource_signal)
        decision["resource_signal"] = dict(resource_signal)
        if self.config.metrics_enabled:
            for key, value in resource_signal.items():
                metrics[f"resource_{key}"] = float(value)
        unique_errors = list(dict.fromkeys(cycle_errors)) if cycle_errors else []
        if unique_errors:
            decision["errors"] = list(unique_errors)

        energy_used_int = int(round(energy_used))
        result = BrainCycleResult(
            perception=perception_snapshot,
            emotion=emotion_snapshot,
            intent=intent,
            personality=personality_snapshot,
            curiosity=curiosity_snapshot,
            energy_used=energy_used_int,
            idle_skipped=int(idle_skipped),
            thoughts=thought_snapshot,
            feeling=feeling_snapshot,
            metrics=metrics,
            metadata={
                "plan": plan.describe(),
                "executed_action": str(action),
                "cognitive_plan": ",".join(decision.get("plan", []))
                if self.config.enable_plan_logging
                else None,
                "memory_size": str(len(self.cognition.episodic_memory)),
                "context_task": str(cognitive_context.get("task"))
                if cognitive_context.get("task") is not None
                else None,
                "oscillation_state": str(oscillation_state) if oscillation_state else None,
                "motor_spike_counts": str(motor_result.spike_counts)
                if motor_result
                else None,
                "motor_average_rate": str(motor_result.average_rate)
                if motor_result and motor_result.average_rate
                else None,
                "motor_energy": str(motor_result.energy_used) if motor_result else None,
                "feedback_metrics": dict(feedback_metrics) if feedback_metrics else None,
                "policy": policy_metadata.get("policy"),
                "policy_metadata": policy_metadata or None,
                "cycle_errors": unique_errors or None,
                "semantic_labels": {
                    key: list(value.get("labels", []))
                    for key, value in perception_snapshot.semantic.items()
                }
                if perception_snapshot.semantic
                else None,
                "semantic_summary": {
                    key: value.get("summary")
                    for key, value in perception_snapshot.semantic.items()
                }
                if perception_snapshot.semantic
                else None,
                "topology_functional": json.dumps(topology_snapshot["functional"]),
                "topology_anatomical": json.dumps(topology_snapshot["anatomical"]),
                "topology_layers": json.dumps(topology_snapshot["layers"]),
                "topology_connectome": json.dumps(topology_snapshot["connectome"]),
                "pipeline_routing": routing_plan,
                "resource_signal": resource_signal,
            },
        )
        self.last_perception = perception_snapshot
        self.last_context = cognitive_context
        self.last_learning_prediction = learning_prediction
        self.last_decision = decision
        self.perception_history.append(perception_snapshot)
        self.decision_history.append(dict(decision))
        cycle_telemetry.update(
            {
                "intention": intention,
                "confidence": decision.get("confidence"),
                "policy": policy_metadata.get("policy"),
            }
        )
        cycle_telemetry["topology_layers"] = dict(topology_snapshot["layers"])
        cycle_telemetry["topology_functional"] = dict(module_activity)
        if unique_errors:
            cycle_telemetry["errors"] = unique_errors
        if plan:
            cycle_telemetry["plan_length"] = len(plan.stages)
        if decision.get("plan"):
            cycle_telemetry["cognitive_plan"] = list(decision.get("plan", []))
        if perception_snapshot.semantic:
            cycle_telemetry["semantic"] = {
                key: {
                    "labels": list(value.get("labels", [])),
                    "summary": value.get("summary"),
                }
                for key, value in perception_snapshot.semantic.items()
            }
        self.telemetry_log.append(dict(cycle_telemetry))
        try:
            emotion_primary = (
                emotion_snapshot.primary.value
                if hasattr(emotion_snapshot.primary, "value")
                else str(emotion_snapshot.primary)
            )
            emotion_meta = {
                "primary": emotion_primary,
                "intensity": float(emotion_snapshot.intensity),
                "mood": float(emotion_snapshot.mood),
                "valence": float(emotion_snapshot.dimensions.get("valence", emotion_snapshot.mood)),
                "arousal": float(
                    emotion_snapshot.dimensions.get("arousal", abs(emotion_snapshot.mood))
                ),
                "confidence": max(0.0, min(1.0, 1.0 - float(emotion_snapshot.decay))),
            }
            assumptions = None
            if isinstance(decision.get("assumptions"), (dict, list, tuple, set)):
                assumptions = decision.get("assumptions")
            elif isinstance(input_data.get("assumptions"), (dict, list, tuple, set)):
                assumptions = input_data.get("assumptions")
            executed_strategy = decision.get("selected_strategy") or (
                plan.describe() if plan else intention
            )
            action_success = action.success if isinstance(action, MotorExecutionResult) else None
            self.self_model.record_cycle(
                goal=cognitive_context.get("task") or decision.get("goal") or intention,
                assumptions=assumptions,
                strategies=decision.get("weights"),
                executed_strategy=executed_strategy,
                success=action_success,
                reward=(feedback_metrics or {}).get("reward") if feedback_metrics else reward_signal,
                feedback=feedback_metrics,
                context=cognitive_context,
                errors=unique_errors,
                capabilities=capability_display_signals,
                emotion=emotion_meta,
                curiosity_drive=self.curiosity.drive,
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.debug("Self-model update failed.", exc_info=True)

        if self.continual_learning is not None and self.config.enable_continual_learning:
            try:
                reward_value = 0.0
                if feedback_metrics and "reward" in feedback_metrics:
                    reward_value = float(feedback_metrics.get("reward", 0.0) or 0.0)
                elif reward_signal is not None:
                    reward_value = float(reward_signal)

                success_value: bool
                if feedback_metrics and "success" in feedback_metrics:
                    success_value = bool(feedback_metrics.get("success"))
                elif isinstance(action, MotorExecutionResult):
                    success_value = bool(action.success)
                else:
                    success_value = reward_value > 0.0

                encoder = BanditCognitivePolicy(fallback=self.cognition.policy)
                state = encoder._encode_state(  # type: ignore[attr-defined]
                    dict(perception_summary) if isinstance(perception_summary, dict) else {},
                    emotion_snapshot,
                    self.curiosity,
                    cognitive_context,
                    learning_prediction if isinstance(learning_prediction, dict) else None,
                )
                state_key = encoder._state_key(state)  # type: ignore[attr-defined]

                task_id = cognitive_context.get("task") or decision.get("goal") or intention
                if task_id is None:
                    task_id = f"cycle:{self.cycle_index}"
                policy_version = str(policy_metadata.get("policy") or type(self.cognition.policy).__name__)

                root = Path(self.config.continual_learning_experience_root)
                trajectory_dir = root / "trajectories"
                trajectory_dir.mkdir(parents=True, exist_ok=True)
                trajectory_path = trajectory_dir / f"cycle_{self.cycle_index}_{int(time.time())}.json"

                trajectory_payload = {
                    "cycle_index": int(self.cycle_index),
                    "task_id": str(task_id),
                    "intention": str(intention),
                    "plan": list(decision.get("plan", [])) if isinstance(decision.get("plan"), list) else [],
                    "reward": float(reward_value),
                    "success": bool(success_value),
                    "timestamp": time.time(),
                    "features": {
                        "state_key": state_key,
                        "focus": max(perception_summary, key=perception_summary.get) if perception_summary else None,
                        "valence": float(emotion_snapshot.dimensions.get("valence", 0.0)),
                        "arousal": float(emotion_snapshot.dimensions.get("arousal", 0.5)),
                        "novelty": float(self.curiosity.last_novelty),
                        "threat": float(cognitive_context.get("threat", 0.0) or 0.0),
                        "safety": float(cognitive_context.get("safety", 0.0) or 0.0),
                    },
                }
                if isinstance(learning_prediction, dict) and learning_prediction:
                    trajectory_payload["features"]["cpu"] = float(learning_prediction.get("cpu", 0.0) or 0.0)
                    trajectory_payload["features"]["memory"] = float(learning_prediction.get("memory", 0.0) or 0.0)

                trajectory_path.write_text(
                    json.dumps(trajectory_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                self.continual_learning.register_episode(
                    str(task_id),
                    policy_version,
                    float(reward_value),
                    1,
                    bool(success_value),
                    metadata={
                        "intention": str(intention),
                        "state_key": state_key,
                        "cycle_index": int(self.cycle_index),
                        "policy": policy_version,
                        "confidence": float(decision.get("confidence", 0.0) or 0.0),
                        "knowledge_gap": bool(
                            cognitive_context.get("knowledge_query")
                            and not (
                                (self.last_memory_retrieval or {}).get("records")
                                or (self.last_memory_retrieval or {}).get("known_facts")
                            )
                        ),
                    },
                    trajectory_path=str(trajectory_path),
                )

                if self.knowledge_base is None and perception_snapshot.knowledge_facts:
                    for fact in perception_snapshot.knowledge_facts:
                        self.continual_learning.register_knowledge_fact(fact)
            except Exception:  # pragma: no cover - best-effort logging
                logger.debug("Continual learning episode capture failed.", exc_info=True)

        if self._metrics_collector is not None and self.config.enable_self_evolution:
            try:
                status = None
                if feedback_metrics and "success" in feedback_metrics:
                    status = "success" if bool(feedback_metrics.get("success")) else "failure"
                confidence = None
                try:
                    confidence = float(decision.get("confidence", 0.0))
                except Exception:
                    confidence = None
                self._metrics_collector.end(
                    "whole_brain_cycle",
                    status=status,
                    confidence=confidence,
                    stage=str(intention) if intention is not None else "cycle",
                )
            except Exception:
                pass
        return result

    def configure_semantic_bridge(self, **kwargs) -> Optional[SemanticBridge]:
        """Replace or reconfigure the semantic bridge used for sensory decoding."""

        try:
            self.semantic_bridge = SemanticBridge(**kwargs)
        except Exception:  # pragma: no cover - optional configuration
            logger.debug("Semantic bridge configuration failed.", exc_info=True)
            self.semantic_bridge = None
        return getattr(self, "semantic_bridge", None)

    def update_config(self, config: BrainRuntimeConfig) -> None:
        """Replace runtime configuration and keep derived flags in sync."""

        self.config = config
        self.neuromorphic = config.use_neuromorphic

    def get_decision_trace(self, limit: int = 5) -> List[dict[str, Any]]:
        """Return recent cognitive decisions for inspection."""

        return self.cognition.recall(limit)

    def get_strategy_modulation(self) -> Dict[str, float]:
        """Expose the latest action weights for agent loop adjustments."""

        weights = self.last_decision.get("weights", {}) if isinstance(self.last_decision, dict) else {}
        return {
            "approach": float(weights.get("approach", 0.0)),
            "withdraw": float(weights.get("withdraw", 0.0)),
            "explore": float(weights.get("explore", 0.0)),
            "observe": float(weights.get("observe", 0.0)),
            "curiosity_drive": float(self.curiosity.drive),
        }

__all__ = [
    "WholeBrainSimulation",
    "CognitiveModule",
    "CognitivePolicy",
    "HeuristicCognitivePolicy",
    "ProductionCognitivePolicy",
    "ReinforcementCognitivePolicy",
    "StructuredPlanner",
    "default_plan_for_intention",
]

