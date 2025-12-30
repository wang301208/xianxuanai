"""Hooks for distributed training and inference.

These functions are placeholders that would be extended to integrate with
frameworks like PyTorch's ``DistributedDataParallel`` or Ray.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional


@dataclass
class DistributedContext:
    enabled: bool
    backend: str
    world_size: int
    rank: int
    local_rank: int


_CONTEXT: Optional[DistributedContext] = None


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _maybe_torch_distributed():
    try:  # pragma: no cover - optional dependency
        import torch.distributed as dist  # type: ignore
    except Exception:
        return None
    return dist


def setup_training() -> None:
    """Prepare the distributed environment for training."""
    global _CONTEXT
    backend = os.getenv("DIST_BACKEND", "nccl" if os.getenv("CUDA_VISIBLE_DEVICES") else "gloo")
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", rank)

    dist = _maybe_torch_distributed()
    enabled = world_size > 1 and dist is not None
    _CONTEXT = DistributedContext(
        enabled=enabled,
        backend=backend,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
    )

    if not enabled:
        print(f"Distributed training disabled (world_size={world_size}).")
        return

    if dist.is_initialized():  # pragma: no cover - already configured
        print("Distributed training already initialized.")
        return

    init_method = os.getenv("DIST_INIT_METHOD", "env://")
    try:
        dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)
        print(f"Initialized distributed training backend={backend} rank={rank}/{world_size} init_method={init_method}")
    except Exception as exc:
        _CONTEXT.enabled = False  # type: ignore[misc]
        print(f"Failed to initialize distributed training ({exc}); continuing without distribution.")


def teardown_training() -> None:
    """Clean up distributed training resources."""
    global _CONTEXT
    dist = _maybe_torch_distributed()
    if dist is None or not dist.is_initialized() or _CONTEXT is None or not _CONTEXT.enabled:
        _CONTEXT = None
        print("Distributed training teardown complete.")
        return
    try:
        dist.destroy_process_group()
        print("Distributed training process group destroyed.")
    finally:
        _CONTEXT = None


def setup_inference() -> None:
    """Prepare the distributed environment for inference."""
    # Inference defaults to the same initialization semantics as training.
    setup_training()


def teardown_inference() -> None:
    teardown_training()


__all__ = [
    "DistributedContext",
    "setup_training",
    "teardown_training",
    "setup_inference",
    "teardown_inference",
]
