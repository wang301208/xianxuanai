import sys
from pathlib import Path

import torch
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from backend.ml.models import get_model, MLP, VisionCNN, SequenceRNN


def _train_step(model, x, y):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()
    before = [p.clone() for p in model.parameters()]
    pred = model(x)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    changed = any(not torch.equal(b, a) for b, a in zip(before, model.parameters()))
    with torch.no_grad():
        final_loss = criterion(model(x), y).item()
    return changed, final_loss


def test_mlp_forward_and_train():
    torch.manual_seed(0)
    model = get_model("mlp", input_dim=4, hidden_dims=[8], output_dim=1)
    x = torch.randn(2, 4)
    y = torch.randn(2, 1)
    out = model(x)
    assert out.shape == (2, 1)
    changed, _ = _train_step(model, x, y)
    assert changed


def test_cnn_forward_and_train():
    torch.manual_seed(0)
    model = get_model("cnn", num_classes=5)
    x = torch.randn(2, 3, 32, 32)
    y = torch.randn(2, 5)
    out = model(x)
    assert out.shape == (2, 5)
    changed, _ = _train_step(model, x, y)
    assert changed


@pytest.mark.parametrize("size", [32, 64, 28])
def test_cnn_forward_varied_input_sizes(size):
    """VisionCNN should handle different image sizes."""
    torch.manual_seed(0)
    model = get_model("cnn", num_classes=5)
    x = torch.randn(2, 3, size, size)
    out = model(x)
    assert out.shape == (2, 5)


def test_rnn_forward_and_train():
    torch.manual_seed(0)
    model = get_model("rnn", input_size=6, hidden_size=4)
    x = torch.randn(2, 3, 6)
    y = torch.randn(2, 4)
    out = model(x)
    assert out.shape == (2, 4)
    changed, _ = _train_step(model, x, y)
    assert changed
