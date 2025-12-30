from BrainSimulationSystem.core.stage_manager import CurriculumStageManager


class DummyEvent:
    def __init__(self, latency: float, module: str = "vision", status: str = "success"):
        self.latency = latency
        self.module = module
        self.status = status
        self.prediction = None
        self.actual = None


class DummyExpander:
    def __init__(self):
        self.calls = []

    def auto_expand(self, performance: float, env_feedback=None, threshold: float = 0.5):
        self.calls.append(
            {"performance": performance, "env_feedback": env_feedback, "threshold": threshold}
        )


def test_stage_manager_promotes_when_window_is_good():
    manager = CurriculumStageManager(starting_stage="infant")
    policy = manager.current_stage.runtime_policy
    events = [
        DummyEvent(latency=policy.max_latency_s * 0.5) for _ in range(policy.promotion_window)
    ]

    transition = manager.ingest_events(events)

    assert transition is not None
    assert transition.previous.key == "infant"
    assert transition.current.key != "infant"


def test_stage_manager_triggers_dynamic_expansion_on_bottleneck():
    expander = DummyExpander()
    manager = CurriculumStageManager(starting_stage="infant", expander=expander)
    policy = manager.current_stage.runtime_policy
    slow_events = [
        DummyEvent(latency=policy.bottleneck_latency_s * 1.5, module="hippocampus")
        for _ in range(5)
    ]

    manager.ingest_events(slow_events)

    assert expander.calls, "dynamic expander should be triggered when bottlenecks persist"


def test_current_config_matches_stage_metadata():
    manager = CurriculumStageManager(starting_stage="juvenile")
    config = manager.current_config()

    assert config["metadata"]["stage"] == "juvenile"
