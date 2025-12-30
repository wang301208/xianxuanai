from __future__ import annotations

"""Basic autoencoder module and reconstruction loss."""

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch may be missing
    torch = None  # type: ignore


class Autoencoder(nn.Module):
    """Simple autoencoder container with separate encoder and decoder."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        if torch is None:  # pragma: no cover - runtime dependency check
            raise ImportError("torch is required for Autoencoder")
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
        z = self.encoder(x)
        return self.decoder(z)


def reconstruction_loss(
    original: "torch.Tensor", reconstruction: "torch.Tensor"
) -> "torch.Tensor":
    """Mean squared error reconstruction loss."""

    if torch is None:  # pragma: no cover - runtime dependency check
        raise ImportError("torch is required for reconstruction_loss")
    loss_fn = nn.MSELoss()
    return loss_fn(reconstruction, original)
