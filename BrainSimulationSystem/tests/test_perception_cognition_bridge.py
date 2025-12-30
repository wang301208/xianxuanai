"""Tests bridging perception embeddings into cognitive context."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from BrainSimulationSystem.brain_simulation import BrainSimulation


def test_brain_simulation_passes_perception_embedding_context_into_memory(monkeypatch):
    simulation = BrainSimulation()

    def fake_compute(**_: object):
        return {"modality_weights": {"language": 0.0, "structured": 1.0}}

    monkeypatch.setattr(simulation.attention_controller, "compute", fake_compute)
    monkeypatch.setattr(simulation, "_build_auto_memory_query", lambda *_args, **_kwargs: None)

    captured: dict = {}

    def fake_memory_process(inputs):
        captured["inputs"] = inputs
        return {"stored_id": None, "retrieved": [], "statistics": {}}

    monkeypatch.setattr(simulation.memory, "process", fake_memory_process)

    def fake_perception_process(_inputs):
        return {
            "perception_output": [0.1, 0.2, 0.3],
            "structured": {"summary_embedding": [2.0, 0.0, 0.0], "attention_weight": 0.01},
            "multimodal_fusion": {
                "embedding": [0.25, 0.5, 0.75, 1.0],
                "modalities": ["structured", "language"],
                "attention_weights": {"structured": 0.1, "language": 0.9},
            },
        }

    monkeypatch.setattr(simulation.perception, "process", fake_perception_process)

    simulation.step(
        {
            "sensory_data": [0.2, 0.5, 0.8],
            "attention_directives": {"modality_weights": {"language": 1.0, "structured": 0.01}},
        },
        dt=0.1,
    )

    memory_inputs = captured.get("inputs")
    assert isinstance(memory_inputs, dict)
    assert "context" in memory_inputs

    context = memory_inputs["context"]
    assert isinstance(context, dict)
    assert "perception" in context
    assert "attention_directives" in context

    perception_context = context["perception"]
    assert perception_context.get("concept_embedding_preview") == [0.25, 0.5, 0.75, 1.0]
    assert perception_context.get("modality_weights", {}).get("structured") == 0.01
    assert context.get("attention_directives", {}).get("modality_weights", {}).get("structured") == 0.01
