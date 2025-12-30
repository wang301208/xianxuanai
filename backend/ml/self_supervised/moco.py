from __future__ import annotations

"""Minimal Momentum Contrast (MoCo) implementation."""

from typing import Callable

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch may be missing
    torch = None  # type: ignore


class MoCo(nn.Module):
    """Simplified MoCo model with a queue and momentum encoder."""

    def __init__(
        self,
        encoder_q: nn.Module,
        encoder_k: nn.Module,
        projection_head: nn.Module,
        augmentations: Callable[["torch.Tensor"], "torch.Tensor"] | None = None,
        queue_size: int = 1024,
        momentum: float = 0.999,
        temperature: float = 0.07,
    ) -> None:
        if torch is None:  # pragma: no cover - runtime dependency check
            raise ImportError("torch is required for MoCo")
        super().__init__()
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        self.projector = projection_head
        self.augment = augmentations or (lambda x: x)
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        self.register_buffer("queue", torch.randn(projection_head.out_features, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: "torch.Tensor") -> None:
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:  # type: ignore[override]
        q = self.augment(x)
        k = self.augment(x)
        q = F.normalize(self.projector(self.encoder_q(q)), dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = F.normalize(self.projector(self.encoder_k(k)), dim=1)
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        self._dequeue_and_enqueue(k)
        return logits, labels
