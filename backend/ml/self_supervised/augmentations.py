from __future__ import annotations

"""Simple augmentation utilities for self-supervised learning."""

from typing import Callable, Iterable

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - torch may be missing
    torch = None  # type: ignore


class AugmentationPipeline:
    """A callable container for a sequence of augmentations."""

    def __init__(self, augmentations: Iterable[Callable[["torch.Tensor"], "torch.Tensor"]] | None = None) -> None:
        self.augmentations = list(augmentations or [])

    def __call__(self, x: "torch.Tensor") -> "torch.Tensor":
        for aug in self.augmentations:
            x = aug(x)
        return x

    def add(self, augmentation: Callable[["torch.Tensor"], "torch.Tensor"]) -> None:
        self.augmentations.append(augmentation)


def random_horizontal_flip(p: float = 0.5) -> Callable[["torch.Tensor"], "torch.Tensor"]:
    """Return an augmentation that randomly flips images horizontally."""

    if torch is None:  # pragma: no cover - runtime dependency check
        raise ImportError("torch is required for random_horizontal_flip")

    def _augment(x: "torch.Tensor") -> "torch.Tensor":
        if torch.rand(1) < p:
            return torch.flip(x, dims=[-1])
        return x

    return _augment
