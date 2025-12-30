import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "third_party/autogpt"))
sys.path.insert(0, str(ROOT / "modules"))

from third_party.autogpt.autogpt.core.brain.config import TransformerBrainConfig
from third_party.autogpt.autogpt.core.brain.transformer_brain import TransformerBrain
from third_party.autogpt.autogpt.core.brain.train_transformer_brain import (
    ObservationActionDataset,
    load_brain,
    save_brain,
    train,
)


def test_training_updates_model(tmp_path):
    config = TransformerBrainConfig(epochs=1, learning_rate=0.01, batch_size=8)
    brain = TransformerBrain(config)
    dataset = ObservationActionDataset(32, config.dim)

    initial = {k: v.clone() for k, v in brain.state_dict().items()}
    train(brain, dataset)

    updated = brain.state_dict()
    assert any(not torch.equal(initial[k], updated[k]) for k in initial)

    save_path = tmp_path / "brain.pth"
    save_brain(brain, save_path)
    reloaded = load_brain(config, save_path)
    for p1, p2 in zip(brain.parameters(), reloaded.parameters()):
        assert torch.equal(p1, p2)
