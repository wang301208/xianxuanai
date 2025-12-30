from BrainSimulationSystem.curriculum import (
    CurriculumManager,
    StageDefinition,
    StageObjective,
    StageTask,
)
from BrainSimulationSystem.config.stage_profiles import build_stage_config


class StubCoordinator:
    def __init__(self) -> None:
        self.state = {}

    def set_loop_enabled(self, name: str, enabled: bool) -> None:
        self.state[name] = bool(enabled)


def _make_stage(stage_key: str, min_rate: float, min_reward: float, loops=(), required_assessments: int = 0):
    return StageDefinition(
        stage_key=stage_key,
        label=stage_key,
        tasks=[StageTask(name=f"{stage_key}-task", environment_id="sim:test")],
        objective=StageObjective(
            min_success_rate=min_rate,
            min_average_reward=min_reward,
            min_episodes=2,
            evaluation_window=4,
            required_assessments=required_assessments,
        ),
        enable_loops=loops,
    )


def test_curriculum_advances_and_enables_loops():
    coordinator = StubCoordinator()
    stages = [
        _make_stage("infant", 0.5, 0.5, loops=("skills",), required_assessments=0),
        _make_stage("juvenile", 0.7, 0.8, loops=("skills", "knowledge"), required_assessments=0),
    ]
    curriculum = CurriculumManager(stages, coordinator=coordinator)

    first = curriculum.record_outcome(success=True, reward=1.0)
    advanced = curriculum.record_outcome(success=False, reward=0.2)

    assert first is False
    assert advanced is True
    assert curriculum.current_stage.stage_key == "juvenile"
    assert coordinator.state["skills"] is True
    assert coordinator.state["knowledge"] is True


def test_stage_config_builds_from_profile():
    curriculum = CurriculumManager([_make_stage("infant", 0.5, 0.5)])
    config = curriculum.current_stage_config()

    assert config["metadata"]["stage"] == "infant"
    assert "brain_regions" in config


def test_record_assessment_advances_stage():
    curriculum = CurriculumManager(
        [
            _make_stage("infant", 0.5, 0.5, required_assessments=1),
            _make_stage("juvenile", 0.6, 0.7),
        ]
    )
    progressed = curriculum.record_assessment(passed=True, metrics={"score": 0.9})
    assert progressed is True
    assert curriculum.current_stage.stage_key == "juvenile"
