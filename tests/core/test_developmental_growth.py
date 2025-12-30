from BrainSimulationSystem.brain_simulation import BrainSimulation
from BrainSimulationSystem.core.developmental_growth import DevelopmentalGrowthController
from BrainSimulationSystem.core.stage_manager import CurriculumStageManager
from BrainSimulationSystem.models.symbolic_reasoner import Rule


class DummyEvent:
    def __init__(self, latency: float, module: str = "task", status: str = "success"):
        self.latency = latency
        self.module = module
        self.status = status
        self.prediction = None
        self.actual = None


class StubSimulation:
    def __init__(self) -> None:
        self.calls = []

    def upgrade_stage(self, stage: str, *, base_profile=None, preserve_state: bool = True, overrides=None) -> None:
        self.calls.append(
            {
                "stage": stage,
                "base_profile": base_profile,
                "preserve_state": preserve_state,
                "overrides": overrides,
            }
        )


def test_growth_controller_applies_stage_transition_to_simulation():
    manager = CurriculumStageManager(starting_stage="infant")
    policy = manager.current_stage.runtime_policy
    events = [DummyEvent(latency=policy.max_latency_s * 0.5) for _ in range(policy.promotion_window)]

    simulation = StubSimulation()
    controller = DevelopmentalGrowthController(simulation=simulation, stage_manager=manager)

    transition = controller.ingest_events(events)

    assert transition is not None
    assert simulation.calls, "expected upgrade_stage to be called"
    assert simulation.calls[-1]["stage"] == transition.current.key
    assert simulation.calls[-1]["base_profile"] == transition.current.base_profile


def test_brain_simulation_upgrade_stage_preserves_symbolic_knowledge_and_grows_network():
    simulation = BrainSimulation(stage="infant")
    initial_columns = len(getattr(simulation.network, "cortical_columns", {}) or {})

    simulation.memory.process(
        {
            "store": {
                "memory_type": "EPISODIC",
                "content": {"concept": "apple", "event": "unit_test"},
                "context": {"source": "test"},
            }
        }
    )

    simulation.knowledge_graph.add("A", "supports", "B")
    simulation.symbolic_reasoner.add_rule(
        Rule(
            name="rule-1",
            antecedents=[("A", "supports", "B")],
            consequent=("A", "related_to", "B"),
        )
    )

    simulation.upgrade_stage("juvenile", preserve_state=True)

    assert simulation.config["metadata"]["stage"] == "juvenile"
    assert simulation.knowledge_graph.exists("A", "supports", "B") is True
    assert "rule-1" in simulation.symbolic_reasoner.rules
    assert len(getattr(simulation.network, "cortical_columns", {}) or {}) > initial_columns

    ca1 = simulation.memory.memory_system.hippocampal_system.ca1_memories
    assert any(
        isinstance(trace.content, dict) and trace.content.get("concept") == "apple"
        for trace in ca1.values()
    )
