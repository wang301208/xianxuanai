"""Learning stack utilities for reinforcement and continual learning."""

from .experience_hub import EpisodeRecord, DemonstrationRecord, ExperienceHub
from .continual_learning import ContinualLearningCoordinator, LearningLoopConfig
from .imitation import ImitationLearner, ImitationStats
from .representation import RepresentationLearner, RepresentationStats
from modules.meta_cognition import MetaCognitionController, MetaSignal
from modules.autonomy import AutonomousTaskGenerator, GeneratedTask, MemoryConsolidator
from modules.task_scheduler import TaskScheduler, ScheduledTask
from .meta_retrieval_policy import MetaRetrievalPolicy, MetaRetrievalPolicyConfig
from .code_doc_self_supervised import CodeDocDatasetConfig, build_self_supervised_examples, write_jsonl
from .learning_agent import LearningAgent, LearningAgentConfig, run_rl_training
from .behavior_cloning import BehaviorCloningConfig, BehaviorCloningPolicy

try:  # Optional dependency on torch via policy head
    from .policy_head import AutoGPTPolicyHead, PolicyConfig
except Exception:  # pragma: no cover - optional dependency may be absent
    AutoGPTPolicyHead = None  # type: ignore[assignment]
    PolicyConfig = None  # type: ignore[assignment]

try:
    from .agent_reflector import AgentReflector, build_reflection_callback
except Exception:  # pragma: no cover - optional during minimal installs
    AgentReflector = None  # type: ignore[assignment]
    build_reflection_callback = None  # type: ignore[assignment]

__all__ = [
    "EpisodeRecord",
    "ExperienceHub",
    "DemonstrationRecord",
    "AutoGPTPolicyHead",
    "PolicyConfig",
    "ContinualLearningCoordinator",
    "LearningLoopConfig",
    "ImitationLearner",
    "ImitationStats",
    "RepresentationLearner",
    "RepresentationStats",
    "MetaCognitionController",
    "MetaSignal",
    "AutonomousTaskGenerator",
    "GeneratedTask",
    "MemoryConsolidator",
    "TaskScheduler",
    "ScheduledTask",
    "AgentReflector",
    "build_reflection_callback",
    "MetaRetrievalPolicy",
    "MetaRetrievalPolicyConfig",
    "CodeDocDatasetConfig",
    "build_self_supervised_examples",
    "write_jsonl",
    "LearningAgent",
    "LearningAgentConfig",
    "run_rl_training",
    "BehaviorCloningConfig",
    "BehaviorCloningPolicy",
]
