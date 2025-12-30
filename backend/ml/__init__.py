"""AutoGPT machine learning utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .backends import get_backend
from .hf_models import load_bert, load_gpt, load_vit
from .models import (
    MLP,
    SequenceRNN,
    TransformerTextModel,
    VisionCNN,
    get_model,
)
try:  # Optional heavy dependencies (torch, transformers, etc.)
    from .learning_loop import (
        ExecutionFeedback,
        LearningCycleConfig,
        LearningLoopOrchestrator,
        ReplayConfig,
    )
except Exception:  # pragma: no cover - fallback when optional deps missing
    LearningLoopOrchestrator = None  # type: ignore[assignment]
    LearningCycleConfig = None  # type: ignore[assignment]
    ExecutionFeedback = None  # type: ignore[assignment]
    ReplayConfig = None  # type: ignore[assignment]

try:
    from .workspace_bridge import WorkspaceBridge, WorkspaceEvent
except Exception:  # pragma: no cover - optional dependency chain
    WorkspaceBridge = None  # type: ignore[assignment]
    WorkspaceEvent = None  # type: ignore[assignment]

try:
    from .fusion import CrossModuleFusion
except Exception:  # pragma: no cover - optional dependency chain
    CrossModuleFusion = None  # type: ignore[assignment]

from .evolutionary_learner import EvolutionaryLearner, EvolutionaryPopulation, SearchDimension


@dataclass
class TrainingConfig:
    """Configuration for incremental training.

    This configuration now supports multiple optimizers, optional learning
    rate scheduling and an early stopping mechanism. Hooks for adversarial and
    curriculum learning strategies can be toggled via this configuration as
    well. Additional flags enable strategies to mitigate catastrophic
    forgetting such as EWC and orthogonal gradient descent. The
    ``task_model_types`` mapping allows assigning different model
    architectures to individual training tasks.
    """

    initial_lr: float = 1e-4
    optimizer: str = "adam"
    lr_scheduler: str | None = None
    early_stopping_patience: int | None = None
    use_adversarial: bool = False
    use_curriculum: bool = False
    use_ewc: bool = False
    use_orthogonal: bool = False
    train_after_samples: int = 100
    checkpoint_dir: Path = Path("data") / "checkpoints"
    model_type: str = "mlp"
    task_model_types: Dict[str, str] | None = None


DEFAULT_TRAINING_CONFIG = TrainingConfig()

__all__ = [
    "get_backend",
    "get_model",
    "TrainingConfig",
    "DEFAULT_TRAINING_CONFIG",
    "load_gpt",
    "load_bert",
    "load_vit",
    "MLP",
    "TransformerTextModel",
    "VisionCNN",
    "SequenceRNN",
    "LearningLoopOrchestrator",
    "LearningCycleConfig",
    "ExecutionFeedback",
    "ReplayConfig",
    "WorkspaceBridge",
    "WorkspaceEvent",
    "CrossModuleFusion",
    "EvolutionaryLearner",
    "EvolutionaryPopulation",
    "SearchDimension",
]
