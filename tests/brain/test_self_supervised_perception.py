import numpy as np

from modules.perception.semantic_bridge import SemanticBridge


class _Snapshot:
    def __init__(self, modalities):
        self.modalities = modalities


def test_semantic_bridge_adds_self_supervised_annotations():
    bridge = SemanticBridge(self_supervised_config={"max_dim": 8})
    snapshot = _Snapshot(
        {
            "vision": {"embedding": np.array([1.0, 0.0, 0.0, 0.0])},
            "audio": {"embedding": np.array([0.0, 1.0, 0.0, 0.0])},
        }
    )

    output = bridge.process(snapshot, ingest=False)

    # Modality annotations get prediction error stats.
    vision_ss = output.semantic_annotations["vision"]["self_supervised"]
    audio_ss = output.semantic_annotations["audio"]["self_supervised"]
    assert "prediction_error" in vision_ss
    assert "prediction_error" in audio_ss

    # Multimodal annotation receives contrastive metrics and a concept key.
    multimodal_ss = output.semantic_annotations["multimodal"]["self_supervised"]
    assert "contrastive_loss" in multimodal_ss
    assert output.semantic_annotations["multimodal"].get("concept_key") is not None

    # Fused embedding and concept knowledge fact should be present.
    assert output.fused_embedding is not None
    assert any(fact.get("predicate") == "hasConceptEmbedding" for fact in output.knowledge_facts)
