import pytest

from modules.autonomy import (
    AgentProfile,
    CognitiveEngineBridge,
    GoalRefinementLoop,
    PerceptionRouter,
    RoleBasedAgentOrchestrator,
)


class _DummyRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, dict]] = []

    def invoke(self, name: str, params: dict, **context: dict) -> dict:
        self.calls.append((name, params, context))
        return {"name": name, "params": params, "context": context}


def test_cognitive_engine_bridge_executes_planned_actions() -> None:
    registry = _DummyRegistry()

    def planner(goal: str, ctx: dict) -> list[dict]:
        return [
            {"name": "plan", "skill": "search", "parameters": {"query": goal}},
            {"name": "review", "skill": "summarize", "parameters": {"audience": ctx.get("user")}},
        ]

    bridge = CognitiveEngineBridge(planner=planner, skill_registry=registry)
    results = bridge.execute_plan("find docs", {"user": "operator"})

    assert [entry[0] for entry in registry.calls] == ["search", "summarize"]
    assert results[0]["parameters"] == {"query": "find docs"}
    assert results[1]["parameters"] == {"audience": "operator"}


def test_perception_router_prioritises_and_deduplicates_signals() -> None:
    def high_priority(_: dict) -> list[dict]:
        return [
            {
                "type": "web",
                "priority": 0.9,
                "source": "browser",
                "payload": {"url": "example.com"},
            },
            {
                "type": "web",
                "priority": 0.1,
                "source": "browser",
                "payload": {"url": "stale"},
            },
        ]

    def low_priority(_: dict) -> list[dict]:
        return [
            {"type": "sensor", "priority": 0.2, "source": "thermostat", "payload": {"temp": 25}},
        ]

    router = PerceptionRouter([high_priority, low_priority])
    ordered = router.poll()

    assert ordered[0]["type"] == "web"
    assert ordered[0]["payload"]["url"] == "example.com"
    assert ordered[1]["type"] == "sensor"
    assert router.history(limit=1)[0]["type"] == ordered[-1]["type"]


class _RecordingBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def publish(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


def test_role_based_orchestrator_assigns_to_matching_role() -> None:
    bus = _RecordingBus()
    orchestrator = RoleBasedAgentOrchestrator(event_bus=bus)
    orchestrator.register_agent("planner-1", "planner", specialties=["planning", "analysis"])
    orchestrator.register_agent("builder-1", "builder", specialties=["coding"])

    task = {"role": "planner", "tags": ["analysis"], "description": "Design approach"}
    assigned = orchestrator.assign_task(task)

    assert isinstance(assigned, AgentProfile)
    assert assigned.agent_id == "planner-1"
    assert bus.events[0][0] == "agent.task.assigned"
    assert bus.events[0][1]["agent_id"] == "planner-1"
    assert assigned.inbox[0]["description"] == "Design approach"


def test_goal_refinement_loop_generates_follow_up_tasks() -> None:
    loop = GoalRefinementLoop()
    generated = loop.cycle(
        completed=[{"loss_before": 1.0, "loss_after": 0.25}],
        environment_signals=["webhook-update"],
    )

    assert generated, "should produce follow-up tasks"
    assert any(task.reason in {"knowledge_gap", "environment_change"} for task in generated)
    assert loop.backlog()
