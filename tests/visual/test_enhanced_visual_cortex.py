import os, sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


from modules.brain.visual.enhanced_visual_cortex import EnhancedVisualCortex


def test_enhanced_visual_cortex_pipeline():
    image = np.ones((2, 2))
    cortex = EnhancedVisualCortex()
    outputs = cortex.process_with_attention(image)

    expected_v1 = image * 0.5
    expected_v2 = expected_v1 + 1
    expected_v4 = expected_v2 + 2
    expected_mt = expected_v2.mean()

    assert np.array_equal(outputs["v1"], expected_v1)
    assert np.array_equal(outputs["v2"], expected_v2)
    assert np.array_equal(outputs["v4"], expected_v4)
    assert outputs["mt"] == expected_mt


def test_attention_modulation_optional():
    image = np.ones((2, 2))
    attention_map = np.full_like(image, 2)
    cortex = EnhancedVisualCortex()

    no_attention = cortex.process_with_attention(image)
    with_attention = cortex.process_with_attention(image, attention_map)

    expected_v1_with_attention = (image * 0.5) * attention_map

    assert np.array_equal(with_attention["v1"], expected_v1_with_attention)
    assert not np.array_equal(no_attention["v1"], with_attention["v1"])


def test_enhanced_visual_cortex_output_keys():
    image = np.zeros((1, 1))
    cortex = EnhancedVisualCortex()
    outputs = cortex.process_with_attention(image)
    assert set(outputs.keys()) == {"v1", "v2", "v4", "mt"}
