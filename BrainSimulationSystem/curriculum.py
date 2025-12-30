"""Stage curriculum orchestration for multi-phase training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Sequence

from BrainSimulationSystem.config.stage_profiles import build_stage_config

try:  # Optional import; only needed when coordinating continual learning loops.
    from modules.learning import ContinualLearningCoordinator
except Exception:  # pragma: no cover - optional during lightweight testing
    ContinualLearningCoordinator = None  # type: ignore[assignment]


@dataclass(frozen=True)
class StageObjective:
    min_success_rate: float
    min_average_reward: float
    min_episodes: int
    evaluation_window: int
    required_assessments: int = 0


@dataclass(frozen=True)
class StageTask:
    name: str
    environment_id: str
    description: str = ""
    episodes: int = 0


@dataclass(frozen=True)
class StageDefinition:
    stage_key: str  # Matches stage_profiles (infant, juvenile, etc.)
    label: str
    objective: StageObjective
    tasks: Sequence[StageTask] = field(default_factory=tuple)
    enable_loops: Sequence[str] = field(default_factory=tuple)
    disable_loops: Sequence[str] = field(default_factory=tuple)
    metadata: Dict[str, str] = field(default_factory=dict)


class CurriculumManager:
    """Track curriculum progression and stage-specific configuration."""

    def __init__(
        self,
        stages: Sequence[StageDefinition],
        *,
        coordinator: Optional["ContinualLearningCoordinator"] = None,
        start_index: int = 0,
        evaluator: Optional["DevelopmentalEvaluator"] = None,
    ) -> None:
        if not stages:
            raise ValueError("At least one stage definition is required")
        self._stages: List[StageDefinition] = list(stages)
        self._index = max(0, min(start_index, len(self._stages) - 1))
        self._history: Deque[Dict[str, float]] = deque(maxlen=self.current_stage.objective.evaluation_window)
        self._coordinator = coordinator
        self._evaluator = evaluator
        self._completed = False
        self._assessment_passes = 0
        self._apply_stage_loops(self.current_stage)

    # ------------------------------------------------------------------ #
    @property
    def current_stage(self) -> StageDefinition:
        return self._stages[self._index]

    @property
    def is_complete(self) -> bool:
        return self._completed

    # ------------------------------------------------------------------ #
    def record_outcome(
        self,
        *,
        success: bool,
        reward: float,
        extra: Optional[Mapping[str, float]] = None,
    ) -> bool:
        """Record an episode outcome and return True when stage advances."""

        self._history.append(
            {"success": float(success), "reward": float(reward), **(extra or {})}
        )
        return self._check_progression()

    # ------------------------------------------------------------------ #
    def current_stage_config(self, overrides: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        """Return the merged stage configuration for the active stage."""

        return build_stage_config(
            self.current_stage.stage_key,
            overrides=overrides or {},
        )

    # ------------------------------------------------------------------ #
    def advance_stage(self) -> Optional[StageDefinition]:
        if self._completed or self._index >= len(self._stages) - 1:
            self._completed = True
            return None
        self._index += 1
        self._history.clear()
        stage = self.current_stage
        self._history = deque(maxlen=stage.objective.evaluation_window)
        self._assessment_passes = 0
        self._apply_stage_loops(stage)
        return stage

    # ------------------------------------------------------------------ #
    def _check_progression(self) -> bool:
        stage = self.current_stage
        objective = stage.objective
        if len(self._history) < max(objective.min_episodes, objective.evaluation_window // 2):
            return False
        successes = sum(item["success"] for item in self._history)
        rewards = sum(item["reward"] for item in self._history)
        success_rate = successes / len(self._history)
        avg_reward = rewards / len(self._history)
        if (
            success_rate >= objective.min_success_rate
            and avg_reward >= objective.min_average_reward
        ):
            self.advance_stage()
            return True
        return False

    # ------------------------------------------------------------------ #
    def _apply_stage_loops(self, stage: StageDefinition) -> None:
        if self._coordinator is None:
            return
        for loop in stage.disable_loops:
            try:
                self._coordinator.set_loop_enabled(loop, False)
            except Exception:
                continue
        for loop in stage.enable_loops:
            try:
                self._coordinator.set_loop_enabled(loop, True)
            except Exception:
                continue

    # ------------------------------------------------------------------ #
    def record_assessment(
        self,
        *,
        passed: bool,
        stage_key: Optional[str] = None,
        metrics: Optional[Mapping[str, float]] = None,
    ) -> bool:
        """Record the outcome of a developmental assessment."""

        if stage_key and stage_key != self.current_stage.stage_key:
            return False
        if not passed:
            return False
        self._assessment_passes += 1
        required = self.current_stage.objective.required_assessments
        if required and self._assessment_passes >= required:
            self.advance_stage()
            return True
        return False


def build_default_curriculum() -> List[StageDefinition]:
    """Return the default five-stage curriculum."""

    return [
        StageDefinition(
            stage_key="infant",
            label="Stage 0 · Sensorimotor Bootstrapping",
            tasks=[
                StageTask(name="visual-motor-alignment", environment_id="sandbox:reach"),
                StageTask(name="audio-reflex", environment_id="sandbox:tones"),
            ],
            objective=StageObjective(
                min_success_rate=0.65,
                min_average_reward=0.5,
                min_episodes=40,
                evaluation_window=50,
            ),
            enable_loops=("skills",),
            metadata={"modules": "perception,motor"},
        ),
        StageDefinition(
            stage_key="juvenile",
            label="Stage 1 · Spatial Reasoning",
            tasks=[
                StageTask(name="maze-navigation", environment_id="sim:nav_v1"),
                StageTask(name="object-tracking", environment_id="sim:tracking"),
            ],
            objective=StageObjective(
                min_success_rate=0.7,
                min_average_reward=0.8,
                min_episodes=60,
                evaluation_window=80,
            ),
            enable_loops=("skills",),
        ),
        StageDefinition(
            stage_key="adolescent",
            label="Stage 2 · Knowledge Formation",
            tasks=[
                StageTask(name="episodic-memory", environment_id="sim:memory_v1"),
                StageTask(name="causal-puzzles", environment_id="sim:causal_lab"),
            ],
            objective=StageObjective(
                min_success_rate=0.75,
                min_average_reward=1.0,
                min_episodes=80,
                evaluation_window=100,
            ),
            enable_loops=("skills", "knowledge"),
        ),
        StageDefinition(
            stage_key="production",
            label="Stage 3 · Language & Planning",
            tasks=[
                StageTask(name="dialogue-grounding", environment_id="sim:dialogue"),
                StageTask(name="multi-step-plans", environment_id="sim:planner"),
            ],
            objective=StageObjective(
                min_success_rate=0.78,
                min_average_reward=1.2,
                min_episodes=100,
                evaluation_window=120,
            ),
            enable_loops=("skills", "knowledge", "reflection"),
        ),
        StageDefinition(
            stage_key="full",
            label="Stage 4 · Integrated Autonomy",
            tasks=[
                StageTask(name="open-world", environment_id="sim:world"),
                StageTask(name="collaborative-language", environment_id="sim:collab"),
            ],
            objective=StageObjective(
                min_success_rate=0.82,
                min_average_reward=1.4,
                min_episodes=150,
                evaluation_window=150,
            ),
            enable_loops=("skills", "knowledge", "reflection", "architecture"),
        ),
    ]


DEFAULT_CURRICULUM: List[StageDefinition] = build_default_curriculum()


__all__ = [
    "StageObjective",
    "StageTask",
    "StageDefinition",
    "CurriculumManager",
    "DEFAULT_CURRICULUM",
    "build_default_curriculum",
]
