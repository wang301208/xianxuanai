from __future__ import annotations

"""Variational autoencoder utilities."""

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch may be missing
    torch = None  # type: ignore

from .autoencoder import reconstruction_loss


class VariationalAutoencoder(nn.Module):
    """Minimal variational autoencoder with linear layers."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        if torch is None:  # pragma: no cover - runtime dependency check
            raise ImportError("torch is required for VariationalAutoencoder")
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

    def reparameterize(self, mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
        std = torch.exp(0.5 * logvar)
        eps = torch.zeros_like(std)
        return mu + eps * std

    def forward(
        self, x: "torch.Tensor"
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:  # type: ignore[override]
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def kl_divergence(mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
    """KL divergence between a normal distribution and N(0, I)."""

    if torch is None:  # pragma: no cover - runtime dependency check
        raise ImportError("torch is required for kl_divergence")
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
