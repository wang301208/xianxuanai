from __future__ import annotations

"""Minimal SimCLR implementation with pluggable augmentations."""

from typing import Callable

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch may be missing
    torch = None  # type: ignore

class SimCLR(nn.Module):
    """Contrastive learning model following the SimCLR framework."""

    def __init__(
        self,
        encoder: nn.Module,
        projection_head: nn.Module,
        augmentations: Callable[["torch.Tensor"], "torch.Tensor"] | None = None,
        temperature: float = 0.5,
    ) -> None:
        if torch is None:  # pragma: no cover - runtime dependency check
            raise ImportError("torch is required for SimCLR")
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head
        self.augment = augmentations or (lambda x: x)
        self.temperature = temperature

    def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:  # type: ignore[override]
        x_i = self.augment(x)
        x_j = self.augment(x)
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        z_i = self.projection_head(h_i)
        z_j = self.projection_head(h_j)
        return z_i, z_j

    def nt_xent_loss(
        self, z_i: "torch.Tensor", z_j: "torch.Tensor"
    ) -> "torch.Tensor":
        """Compute the NT-Xent contrastive loss."""

        if torch is None:  # pragma: no cover - runtime dependency check
            raise ImportError("torch is required for SimCLR")

        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, dim=1)
        similarity = torch.matmul(z, z.T)
        batch_size = z_i.size(0)
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        similarity = similarity.masked_fill(mask, float("-inf"))
        logits = similarity / self.temperature
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels, labels], dim=0)
        return F.cross_entropy(logits, labels)
