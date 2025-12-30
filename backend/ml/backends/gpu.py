"""GPU backend using CuPy when available."""
from __future__ import annotations

from .base import DeviceBackend


class GPUBackend(DeviceBackend):
    """Backend implementation targeting CUDA GPUs via ``cupy``.

    The backend attempts to minimise host-device transfers by operating on
    device arrays directly and uses a dedicated CUDA stream for kernel
    launches to encourage overlap of transfers and computation.
    """

    name = "gpu"

    def __init__(self) -> None:
        import cupy as cp  # type: ignore

        self.cp = cp
        self.stream = cp.cuda.Stream(non_blocking=True)

    def to_device(self, array):
        with self.stream:
            return self.cp.asarray(array)

    def from_device(self, array):
        with self.stream:
            host = self.cp.asnumpy(array)
        self.stream.synchronize()
        return host

    def matmul(self, a, b):
        with self.stream:
            result = self.cp.matmul(a, b)
        self.stream.synchronize()
        return result
