import os
import sys
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.multimodal.fusion_engine import MultimodalFusionEngine


def _manual_fusion(engine: MultimodalFusionEngine, **modalities):
    aligned, hints = engine._align_modalities(modalities)  # type: ignore[attr-defined]
    weights = engine._attention(aligned, hints)  # type: ignore[attr-defined]
    return np.average(aligned, axis=0, weights=weights)


def test_attention_weighting():
    engine = MultimodalFusionEngine()
    visual = np.ones((2, 2))
    auditory = {"embedding": np.full((2, 2), 10), "confidence": 1.5}
    fused = engine.fuse_sensory_modalities(visual=visual, auditory=auditory)
    expected = _manual_fusion(engine, visual=visual, auditory=auditory)
    assert np.allclose(fused, expected)


def test_support_additional_modalities():
    engine = MultimodalFusionEngine()
    modalities = {
        "visual": np.array([1, 2, 3]),
        "auditory": {"embedding": np.array([4, 5, 6]), "confidence": 1.2},
        "tactile": np.array([7, 8, 9]),
        "smell": np.array([1, 1, 1]),
        "text": np.array([10, 10, 10]),
    }
    fused = engine.fuse_sensory_modalities(**modalities)
    expected = _manual_fusion(engine, **modalities)
    assert np.allclose(fused, expected)


def test_robustness_zero_input():
    engine = MultimodalFusionEngine()
    zero = np.zeros(5)
    one = {"embedding": np.ones(5), "confidence": 1.0}
    fused = engine.fuse_sensory_modalities(zero=zero, one=one)
    expected = _manual_fusion(engine, zero=zero, one=one)
    assert np.allclose(fused, expected)


def test_semantic_bridge_prefers_clip_embedding(monkeypatch, tmp_path):
    from modules.perception import semantic_bridge as semantic_bridge_module

    class _FakeAligner:
        def align(self, embedding, n_results: int, vector_type: str):  # pragma: no cover - simple stub
            return []

    class _FakeClipExtractor:
        def __init__(self, **_: object) -> None:  # pragma: no cover - deterministic stub
            pass

        def extract_image_features(self, image: object) -> np.ndarray:  # pragma: no cover - deterministic stub
            return np.array([0.0, 2.0, 0.0, 0.0], dtype=np.float32)

        def extract_text_features(self, text: str) -> np.ndarray:  # pragma: no cover - deterministic stub
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(semantic_bridge_module, "require_default_aligner", lambda: _FakeAligner())
    monkeypatch.setattr(semantic_bridge_module, "CLIPFeatureExtractor", _FakeClipExtractor)
    monkeypatch.setattr(
        semantic_bridge_module.SemanticBridge,
        "_resolve_clip_image",
        lambda self, payload, metadata: (object(), "payload.image"),
        raising=False,
    )

    snapshot_payload = {
        "vector": [1.0, 0.0, 0.0, 0.0],
        "features": {
            "edge_energy": 0.2,
            "contrast": 0.25,
            "orientation_strength": 0.0,
            "orientation_entropy": 0.9,
        },
    }
    snapshot = SimpleNamespace(modalities={"vision": snapshot_payload})

    bridge_no_clip = semantic_bridge_module.SemanticBridge(storage_root=tmp_path / "no_clip")
    bridge_no_clip._clip_available = False
    bridge_no_clip._clip_extractor = None
    result_no_clip = bridge_no_clip.process(snapshot, ingest=False)

    bridge_with_clip = semantic_bridge_module.SemanticBridge(storage_root=tmp_path / "with_clip")
    result_with_clip = bridge_with_clip.process(snapshot, ingest=False)

    assert "vision" in result_no_clip.modality_embeddings
    assert "vision" in result_with_clip.modality_embeddings

    heuristic_embedding = np.asarray(result_no_clip.modality_embeddings["vision"], dtype=float)
    clip_embedding = np.asarray(result_with_clip.modality_embeddings["vision"], dtype=float)

    assert not np.allclose(heuristic_embedding, clip_embedding)
    assert np.allclose(clip_embedding, np.array([0.0, 1.0, 0.0, 0.0], dtype=float))

    assert result_with_clip.semantic_annotations["vision"]["embedding_source"] == "clip"
    np.testing.assert_allclose(
        result_with_clip.semantic_annotations["vision"]["heuristic_embedding"],
        heuristic_embedding,
    )

    assert np.allclose(result_no_clip.fused_embedding, heuristic_embedding)
    assert np.allclose(result_with_clip.fused_embedding, clip_embedding)


def test_semantic_bridge_text_embedding_contributes(monkeypatch, tmp_path):
    from modules.perception import semantic_bridge as semantic_bridge_module

    class _FakeAligner:
        def align(self, embedding, n_results: int, vector_type: str):  # pragma: no cover - deterministic stub
            return []

    class _FakeClipExtractor:
        def __init__(self, **_: object) -> None:  # pragma: no cover - deterministic stub
            self.tokenizer = lambda texts: np.full((1, 4), 0.5, dtype=np.float32)

        def extract_image_features(self, image: object) -> np.ndarray:  # pragma: no cover - deterministic stub
            return np.ones(4, dtype=np.float32)

        def extract_text_features(self, text: str) -> np.ndarray:  # pragma: no cover - deterministic stub
            return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    monkeypatch.setattr(semantic_bridge_module, "require_default_aligner", lambda: _FakeAligner())
    monkeypatch.setattr(semantic_bridge_module, "CLIPFeatureExtractor", _FakeClipExtractor)

    audio_payload = {
        "vector": [0.4, 0.2, 0.1, 0.3],
        "features": {
            "energy": 0.4,
            "spectral_centroid": 2200.0,
            "spectral_flux": 0.2,
            "temporal_modulation": 0.03,
        },
    }
    text_payload = {"text": "Language observations enrich perception", "confidence": 0.85}

    bridge = semantic_bridge_module.SemanticBridge(
        storage_root=tmp_path / "text",
        asr_config=semantic_bridge_module.ASRConfig(enabled=False),
    )

    snapshot_audio_only = SimpleNamespace(modalities={"audio": audio_payload})
    snapshot_with_text = SimpleNamespace(modalities={"audio": audio_payload, "text": text_payload})

    audio_only_result = bridge.process(snapshot_audio_only, ingest=False)
    combined_result = bridge.process(snapshot_with_text, ingest=False)

    assert "text" in combined_result.semantic_annotations
    text_annotation = combined_result.semantic_annotations["text"]
    assert text_annotation["embedding_source"] == "clip"
    assert "summary" in text_annotation and text_annotation["summary"]

    text_embedding = np.asarray(combined_result.modality_embeddings["text"], dtype=float)
    assert text_embedding.size > 0
    assert np.isclose(np.linalg.norm(text_embedding), 1.0)

    fused_audio_only = np.asarray(audio_only_result.fused_embedding, dtype=float)
    fused_combined = np.asarray(combined_result.fused_embedding, dtype=float)
    assert fused_audio_only.shape == fused_combined.shape
    assert not np.allclose(fused_audio_only, fused_combined)

    predicates = {fact["predicate"] for fact in combined_result.knowledge_facts if fact["subject"].endswith("text:0")}
    assert "hasText" in predicates
    assert "hasSummary" in predicates
