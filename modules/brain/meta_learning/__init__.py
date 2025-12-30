from .core import (
    FewShotTask,
    MetaMemorySystem,
    SelfReflectionModule,
    ReptileOptimizer,
    FewShotAdapter,
    ContinualLearningEngine,
    cross_task_experiment,
    MAMLEngine,
)

from .self_improvement_manager import SelfImprovementManager
from .meta_learning_brain import MetaLearningBrain
from .coordinator import MetaLearningCoordinator, TaskExperience
from .benchmark_coordinator import BenchmarkMetaLearningCoordinator

__all__ = [
    "FewShotTask",
    "MetaMemorySystem",
    "SelfReflectionModule",
    "ReptileOptimizer",
    "FewShotAdapter",
    "ContinualLearningEngine",
    "cross_task_experiment",
    "MAMLEngine",
    "SelfImprovementManager",
    "MetaLearningBrain",
    "MetaLearningCoordinator",
    "TaskExperience",
    "BenchmarkMetaLearningCoordinator",
]
