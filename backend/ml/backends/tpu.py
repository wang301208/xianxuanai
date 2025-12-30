"""TPU backend using JAX when available."""
from __future__ import annotations

from .base import DeviceBackend


class TPUBackend(DeviceBackend):
    """Backend for TPUs using ``jax``.

    JAX handles device placement and asynchronous execution internally, so
    the implementation is straightforward.
    """

    name = "tpu"

    def __init__(self) -> None:
        import jax.numpy as jnp  # type: ignore
        import numpy as np

        self.jnp = jnp
        self.np = np

    def to_device(self, array):
        return self.jnp.asarray(array)

    def from_device(self, array):
        # ``jax`` arrays expose the ``__array__`` protocol so ``numpy`` can
        # transparently transfer them back to host memory.
        return self.np.asarray(array)

    def matmul(self, a, b):
        return self.jnp.matmul(a, b)
