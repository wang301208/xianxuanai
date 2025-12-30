"""Utility helpers for domain adaptation and knowledge accumulation."""

from __future__ import annotations

from typing import Iterable, Sequence

try:  # pragma: no cover - optional runtime dependency
    import torch
    from torch import nn, optim
except Exception:  # pragma: no cover - torch may be missing
    torch = None  # type: ignore
    nn = optim = None  # type: ignore


def fine_tune(
    model: nn.Module,
    dataloader: Iterable[tuple["torch.Tensor", "torch.Tensor"]],
    epochs: int = 1,
    lr: float = 1e-3,
) -> None:
    """Fine-tune ``model`` on a labelled dataset."""

    if torch is None:  # pragma: no cover - runtime dependency check
        raise ImportError("torch is required for fine_tune")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()


def progressive_knowledge_accumulation(
    model: nn.Module,
    datasets: Sequence[Iterable[tuple["torch.Tensor", "torch.Tensor"]]],
    epochs: int = 1,
    lr: float = 1e-3,
) -> None:
    """Sequentially fine-tune ``model`` across multiple datasets."""

    for dl in datasets:
        fine_tune(model, dl, epochs=epochs, lr=lr)
