from __future__ import annotations

from autogpt.core.configuration.schema import (
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)


class LearningConfiguration(SystemConfiguration):
    """Configuration options for experience-based learning."""

    enabled: bool = UserConfigurable(
        default=False, description="Enable learning from stored experiences"
    )
    learning_rate: float = UserConfigurable(
        default=0.001, description="Learning rate for model updates"
    )
    batch_size: int = UserConfigurable(
        default=32, description="Batch size of experiences used per update"
    )
    log_path: str = UserConfigurable(
        default="data/experience_logs.jsonl",
        description="Location of the append-only experience log",
    )
    max_log_bytes: int | None = UserConfigurable(
        default=5_000_000,
        description="Rotate the experience log when it exceeds this size (bytes)",
    )
    max_summary_chars: int = UserConfigurable(
        default=2000,
        description="Truncate recorded outputs to this many characters",
    )
    model_state_path: str = UserConfigurable(
        default="data/experience_model.json",
        description="Where to store learned command weights",
    )
    auto_improve: bool = UserConfigurable(
        default=False,
        description="Automatically analyse experience logs and generate improvement plans",
    )
    improvement_interval: int = UserConfigurable(
        default=5,
        description="Number of cycles between automatic improvement evaluations",
    )
    min_records: int = UserConfigurable(
        default=30,
        description="Minimum number of experience records required before auto-improve runs",
    )
    rollback_tolerance: float = UserConfigurable(
        default=0.05,
        description="If success rate drops by more than this fraction, revert to previous plan",
    )
    improvement_state_path: str = UserConfigurable(
        default="data/self_improvement/state.json",
        description="Where to persist auto-improvement state",
    )
    plan_output_path: str = UserConfigurable(
        default="data/self_improvement/learning_overrides.json",
        description="Latest generated improvement plan",
    )

    ability_history_path: str = UserConfigurable(
        default="data/self_improvement/ability_history.json",
        description="Rolling history of ability-centric evaluation scores",
    )
    ability_low_score_threshold: float = UserConfigurable(
        default=0.6,
        description="Score threshold below which an ability is considered weak",
    )
    ability_low_score_streak: int = UserConfigurable(
        default=3,
        description="Number of consecutive low-score evaluations to flag an ability",
    )
    ability_history_limit: int = UserConfigurable(
        default=50,
        description="Maximum number of historical ability snapshots to persist",
    )


    generate_prompt_candidates: bool = UserConfigurable(
        default=True,
        description="Generate prompt candidate files for review",
    )
    prompt_candidates_dir: str = UserConfigurable(
        default="data/self_improvement/prompt_candidates",
        description="Directory for generated prompt candidates",
    )
    max_prompt_candidates: int = UserConfigurable(
        default=5,
        description="Keep at most this many prompt candidates",
    )


    min_success_improvement: float = UserConfigurable(
        default=0.02,
        description="Minimum improvement in success rate required to accept a plan",
    )
    validation_reports_dir: str = UserConfigurable(
        default="data/self_improvement/reports",
        description="Directory where validation reports are stored",
    )


    baseline_success_path: str = UserConfigurable(
        default="data/self_improvement/baseline.json",
        description="File storing offline baseline success rates",
    )
    replay_scenarios_dir: str = UserConfigurable(
        default="data/self_improvement/replay_scenarios",
        description="Directory containing offline replay scenario definitions",
    )
    replay_reports_dir: str = UserConfigurable(
        default="data/self_improvement/replay_reports",
        description="Directory where replay validation reports are stored",
    )


    protected_commands: list[str] = UserConfigurable(
        default_factory=list,
        description="Commands that may not be disabled by auto improvement",
    )


class LearningSettings(SystemSettings):
    """Settings wrapper for the ExperienceLearner."""

    configuration: LearningConfiguration
