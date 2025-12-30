import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.brain.multimodal import CrossModalTransformer, MultimodalFusionEngine


def test_fused_representation_matches_attention_weights():
    transformer = CrossModalTransformer(output_dim=8)
    engine = MultimodalFusionEngine(transformer)
    modalities = {
        "vision": np.array([1.0, 3.0, 5.0]),
        "audio": {"embedding": np.array([0.2, 0.5, 0.8]), "confidence": 1.5},
        "text": np.array([0.1, 0.4, 0.9, 1.3]),
    }
    fused = engine.fuse_sensory_modalities(**modalities)
    aligned, hints = engine._align_modalities(modalities)  # type: ignore[attr-defined]
    weights = engine._attention(aligned, hints)  # type: ignore[attr-defined]
    expected = np.average(aligned, axis=0, weights=weights)
    assert fused.shape == (transformer.output_dim,)
    np.testing.assert_allclose(fused, expected)


def test_missing_modality_raises_error():
    engine = MultimodalFusionEngine()
    with pytest.raises(ValueError):
        engine.fuse_sensory_modalities()


def test_dependency_injection():
    class DummyTransformer:
        def project(self, array, *, modality=None, metadata=None):
            values = np.asarray(array, dtype=float).reshape(-1)
            return np.full(3, values.sum())

    engine = MultimodalFusionEngine()
    engine.set_transformer(DummyTransformer())
    fused = engine.fuse_sensory_modalities(
        vision=np.array([1.0, 2.0]), text=np.array([3.0])
    )
    assert fused.shape == (3,)
    assert np.allclose(fused, fused[0])


def test_confidence_biases_fusion_towards_stronger_modalities():
    transformer = CrossModalTransformer(output_dim=6)
    engine = MultimodalFusionEngine(transformer)
    text = np.linspace(0.1, 0.6, num=6)
    audio = np.linspace(0.6, 1.1, num=6)
    low_conf = engine.fuse_sensory_modalities(
        text=text,
        audio={"embedding": audio, "confidence": 0.4},
    )
    high_conf = engine.fuse_sensory_modalities(
        text=text,
        audio={"embedding": audio, "confidence": 2.0},
    )
    audio_projection = transformer.project(audio, modality="audio", metadata={"confidence": 2.0})
    low_dist = np.linalg.norm(low_conf - audio_projection)
    high_dist = np.linalg.norm(high_conf - audio_projection)
    assert high_dist < low_dist
