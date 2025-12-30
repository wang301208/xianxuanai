import numpy as np
import pytest

from modules.perception import semantic_bridge
from modules.perception.semantic_bridge import SemanticBridge


class DummySnapshot:
    def __init__(self, payload):
        self.modalities = {"vision": payload}


def make_visual_payload():
    return {
        "vector": [0.1, 0.2, 0.3],
        "features": {
            "edge_energy": 0.5,
            "contrast": 0.4,
            "orientation_strength": 0.5,
            "orientation_entropy": 0.5,
        },
        "metadata": {
            "orientation_histogram": [0.1, 0.9],
            "orientation_angles": [0, 45],
        },
    }


def test_semantic_bridge_enriches_with_clip(monkeypatch):
    class FakeExtractor:
        def __init__(self, **_kwargs):
            pass

        def extract_image_features(self, image):
            assert image == "fake_image"
            return np.array([0.8, 0.2], dtype=np.float32)

        def extract_text_features(self, prompt):
            if prompt == "person prompt":
                return np.array([0.8, 0.2], dtype=np.float32)
            if prompt == "object prompt":
                return np.array([0.2, 0.8], dtype=np.float32)
            return np.ones(2, dtype=np.float32)

    monkeypatch.setattr(semantic_bridge, "CLIPFeatureExtractor", FakeExtractor)
    monkeypatch.setattr(
        SemanticBridge,
        "_resolve_clip_image",
        lambda self, payload, metadata: ("fake_image", "payload.image"),
    )

    bridge = SemanticBridge(
        clip_config={
            "prompts": [("person", "person prompt"), ("object", "object prompt")],
            "top_k": 1,
        }
    )

    snapshot = DummySnapshot(make_visual_payload())

    result = bridge.process(snapshot, agent_id="agent", cycle_index=5, ingest=False)

    annotation = result.semantic_annotations["vision"]
    assert "person" in annotation["labels"]
    assert "clip" in annotation

    clip_info = annotation["clip"]
    assert clip_info["labels"] == ["person"]
    assert clip_info["caption"].startswith("person prompt")
    assert clip_info["source"] == "payload.image"
    assert pytest.approx(np.linalg.norm(clip_info["embedding"]), rel=1e-6) == 1.0

    summary = annotation["summary"]
    assert "CLIP insight" in summary
    assert "person prompt" in summary

    facts = result.knowledge_facts
    assert any(fact["predicate"] == "hasLabel" and fact["obj"] == "person" for fact in facts)

    for fact in facts:
        metadata = fact["metadata"]
        assert metadata["clip_labels"] == ["person"]
        assert metadata["clip_caption"].startswith("person prompt")
        assert metadata["clip_image_source"] == "payload.image"
        assert np.isclose(np.linalg.norm(metadata["clip_embedding"]), 1.0)
        assert metadata["clip_label_scores"]


def test_semantic_bridge_falls_back_without_clip(monkeypatch):
    monkeypatch.setattr(semantic_bridge, "CLIPFeatureExtractor", None)
    monkeypatch.setattr(
        SemanticBridge,
        "_resolve_clip_image",
        lambda self, payload, metadata: ("fake_image", "payload.image"),
    )

    bridge = SemanticBridge()

    snapshot = DummySnapshot(make_visual_payload())
    result = bridge.process(snapshot, agent_id="agent", cycle_index=2, ingest=False)

    annotation = result.semantic_annotations["vision"]
    assert "clip" not in annotation
    assert "CLIP insight" not in annotation["summary"]
    assert "person" not in annotation["labels"]

    for fact in result.knowledge_facts:
        assert all(not key.startswith("clip") for key in fact["metadata"].keys())
