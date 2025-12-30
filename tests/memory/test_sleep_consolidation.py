from BrainSimulationSystem.brain_simulation import BrainSimulation
from BrainSimulationSystem.models.hippocampal_memory import HippocampalMemorySystem
from BrainSimulationSystem.models.memory import ConsolidationState


def test_sleep_consolidate_updates_hippocampal_state_and_ingests_knowledge():
    simulation = BrainSimulation(stage="juvenile")
    result = simulation.memory.process(
        {
            "store": {
                "memory_type": "EPISODIC",
                "content": {
                    "event": "experience",
                    "concept": "apple",
                    "action": "eat",
                    "stage": "juvenile",
                },
                "context": {"source": "test"},
            }
        }
    )
    stored_id = result.get("stored_id")
    trace = simulation.memory.memory_system.hippocampal_system.ca1_memories[stored_id]
    assert trace.consolidation_state == ConsolidationState.LABILE

    report = simulation.sleep_consolidate(duration=100.0, dt=100.0, extract_knowledge=True)

    assert report["steps"] >= 1
    assert trace.consolidation_state != ConsolidationState.LABILE
    assert report["knowledge_triples"] > 0


def test_hippocampal_consolidation_triggers_reconsolidation_for_recent_recall():
    hippocampal = HippocampalMemorySystem({"reconsolidation_window": 3600.0})
    memory_id = hippocampal.encode_episodic_memory({"concept": "apple"}, {})
    trace = hippocampal.ca1_memories[memory_id]

    hippocampal.consolidate_memories(dt=1_000.0, sleep_mode=True)
    hippocampal.consolidate_memories(dt=100_000.0, sleep_mode=True)
    assert trace.consolidation_state == ConsolidationState.CONSOLIDATED

    hippocampal.retrieve_episodic_memory({"concept": "apple"})
    hippocampal.consolidate_memories(dt=1.0, sleep_mode=True)
    assert trace.consolidation_state == ConsolidationState.RECONSOLIDATING

