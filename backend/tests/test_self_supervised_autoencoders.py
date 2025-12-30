import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.ml.self_supervised.autoencoder import Autoencoder, reconstruction_loss
from backend.ml.self_supervised.vae import VariationalAutoencoder

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - skip tests if torch missing
    pytest.skip("torch is required for self-supervised tests", allow_module_level=True)


def test_autoencoder_reconstruction_loss() -> None:
    model = Autoencoder(nn.Identity(), nn.Identity())
    x = torch.randn(2, 3)
    recon = model(x)
    loss = reconstruction_loss(x, recon)
    assert loss.item() == pytest.approx(0.0)


def test_variational_autoencoder_reconstruction_loss() -> None:
    class IdentityVAE(VariationalAutoencoder):
        def __init__(self, dim: int) -> None:
            super().__init__(dim, dim, dim)
            self.encoder = nn.Identity()
            self.fc_mu = nn.Identity()
            self.fc_logvar = nn.Identity()
            self.decoder = nn.Identity()

        def reparameterize(self, mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
            return mu

    x = torch.randn(2, 4)
    model = IdentityVAE(4)
    recon, mu, logvar = model(x)
    loss = reconstruction_loss(x, recon)
    assert loss.item() == pytest.approx(0.0)
