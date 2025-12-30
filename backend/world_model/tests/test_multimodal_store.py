import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


# Mirror the dynamic import pattern used across backend/world_model tests
backend_pkg = types.ModuleType("backend")
backend_pkg.__path__ = []
sys.modules.setdefault("backend", backend_pkg)

wm_path = Path(__file__).resolve().parents[1] / "__init__.py"
spec = importlib.util.spec_from_file_location("backend.world_model", wm_path)
module = importlib.util.module_from_spec(spec)
sys.modules["backend.world_model"] = module
spec.loader.exec_module(module)  # type: ignore[arg-type]

WorldModel = module.WorldModel


def test_multimodal_audio_and_tactile_fusion():
    wm = WorldModel()

    wm.register_modality("audio", encoder=lambda sample: np.asarray(sample, dtype=float))
    wm.register_modality("tactile", encoder=lambda reading: np.asarray(reading, dtype=float))

    waveform = [0.2, 0.4, 0.6, 0.4]
    tactile_vector = [0.1, 0.9, 0.3]

    wm.add_multimodal_observation("agent-a", "audio", data=waveform)
    wm.add_multimodal_observation("agent-a", "tactile", features=tactile_vector)

    unified = wm.get_unified_representation("agent-a")
    assert unified is not None
    assert unified.shape[0] == len(tactile_vector)

    modalities = wm.get_multimodal_observation("agent-a")
    assert set(modalities) == {"audio", "tactile"}
    assert np.allclose(modalities["tactile"]["features"], tactile_vector)

    state = wm.get_state()
    assert "agent-a" in state["multimodal"]
    assert "modalities" in state["multimodal"]["agent-a"]
    assert "audio" in state["multimodal"]["agent-a"]["modalities"]
    assert wm.multimodal["agent-a"] is not None
