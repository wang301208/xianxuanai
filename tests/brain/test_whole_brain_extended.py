import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain import WholeBrainSimulation


class _StubKnowledgeBase:
    def __init__(self) -> None:
        self.facts = []
        self.queries = []

    def ingest_facts(self, facts) -> dict:
        self.facts.extend(facts)
        return {"imported": len(facts)}

    def query(self, concept: str, *, semantic: bool = False, top_k: int = 5) -> dict:
        self.queries.append((concept, semantic, top_k))
        return {"facts": [{"subject": concept, "text": f"{concept} fact"}]}


def test_whole_brain_exposes_oscillation_and_motor_metrics():
    brain = WholeBrainSimulation(neuromorphic=True)
    input_data = {
        "image": [0.7, 0.2, 0.1],
        "sound": [0.5, 0.1],
        "touch": [0.4, 0.4],
        "text": "safety",
        "context": {"task": "explore", "safety": 0.3},
        "is_salient": True,
    }
    result = brain.process_cycle(input_data)

    assert "osc_amplitude" in result.metrics
    assert "osc_synchrony_norm" in result.metrics
    assert result.metrics.get("motor_energy", 0.0) >= 0.0
    assert result.metadata.get("oscillation_state")
    assert result.metadata.get("motor_spike_counts")
    assert "feedback_velocity_error" in result.metrics
    assert "feedback_success_rate" in result.metrics
    assert result.metadata.get("feedback_metrics")
    assert result.metadata.get("policy") == "production"
    assert result.metadata.get("policy_metadata")["confidence_calibrated"] is True
    assert result.metadata.get("cycle_errors") is None
    assert brain.motor.cerebellum is brain.cerebellum
    assert brain.last_motor_result is not None
    assert brain.last_decision.get("motor_spike_counts")
    assert len(brain.decision_history) == 1
    assert brain.telemetry_log[-1]["plan_length"] >= 1


def test_whole_brain_injects_knowledge_context():
    brain = WholeBrainSimulation(neuromorphic=True)
    kb = _StubKnowledgeBase()
    brain.attach_knowledge_base(kb)
    input_data = {
        "text": "sky color inquiry",
        "context": {"task": "describe sky"},
    }
    result = brain.process_cycle(input_data)

    assert kb.queries, "knowledge base should be queried"
    assert "knowledge_context" in brain.last_context
    assert result.metadata.get("policy_metadata", {}).get("knowledge_context")
