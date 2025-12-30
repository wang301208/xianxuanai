from modules.brain.meta_learning.coordinator import MetaLearningCoordinator


class DummyPolicy:
    def __init__(self) -> None:
        self.integrated: list[tuple[float, bool, dict[str, object] | None]] = []
        self.replays = 0

    def integrate_external_feedback(self, reward: float, success: bool, metadata=None) -> None:
        self.integrated.append((reward, success, dict(metadata or {})))

    def replay_from_buffer(self) -> int:
        self.replays += 1
        return self.replays


class StubRegistry:
    def __init__(self) -> None:
        self.registered: list[dict[str, object]] = []

    def register(self, spec, handler=None, replace=False, metadata=None) -> None:  # pragma: no cover - simple stub
        self.registered.append({
            "spec": spec,
            "handler": handler,
            "replace": replace,
            "metadata": dict(metadata or {}),
        })


def test_meta_learning_records_experience_and_updates_policy():
    policy = DummyPolicy()
    coordinator = MetaLearningCoordinator(policy=policy)

    decision = {"intention": "explore", "plan": ["scan_environment", "log_findings"]}
    metrics = {"reward": 0.8, "success": True}

    update = coordinator.record_outcome(
        cycle_index=1,
        state_signature="state-1",
        decision=decision,
        feedback_metrics=metrics,
        reward_signal=metrics["reward"],
        cognitive_context={"task": "survey"},
    )

    assert update is not None
    assert update["intention"] == "explore"
    assert policy.integrated, "policy should receive external feedback"
    assert policy.replays >= 1
    assert coordinator.learned_skills


def test_meta_learning_generates_suggestions_and_registers_skill():
    registry = StubRegistry()
    coordinator = MetaLearningCoordinator(
        policy=None,
        skill_registry=registry,
        success_threshold=0.5,
        min_successes_to_register=1,
    )

    decision = {"intention": "approach", "plan": ["greet", "assist"]}
    metrics = {"reward": 0.6, "success": True}
    coordinator.record_outcome(
        cycle_index=2,
        state_signature="help-request",
        decision=decision,
        feedback_metrics=metrics,
        reward_signal=metrics["reward"],
        cognitive_context={"task": "assist_user"},
    )

    suggestions = coordinator.inject_suggestions({}, perception=None)
    assert suggestions
    assert suggestions[0]["intention"] == "approach"
    assert registry.registered, "successful outcomes should be persisted as skills"

