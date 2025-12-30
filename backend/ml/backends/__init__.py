from .base import DeviceBackend
from .cpu import CPUBackend
from .gpu import GPUBackend
from .tpu import TPUBackend
import os

_BACKENDS = {
    'cpu': CPUBackend,
    'gpu': GPUBackend,
    'tpu': TPUBackend,
}


def get_backend(preferred: str | None = None) -> DeviceBackend:
    """Return a backend instance for the preferred device.

    If the preferred backend is unavailable, gracefully fall back to CPU.
    The ``AUTOGPT_DEVICE`` environment variable can override the preference.
    """
    name = preferred or os.environ.get('AUTOGPT_DEVICE')
    tried: list[str] = []
    if name:
        names = [name.lower()]
    else:
        names = ['gpu', 'tpu', 'cpu']
    for n in names:
        backend_cls = _BACKENDS.get(n)
        if not backend_cls:
            continue
        try:
            return backend_cls()
        except Exception:
            tried.append(n)
            continue
    # last resort
    return CPUBackend()


__all__ = [
    'DeviceBackend',
    'CPUBackend',
    'GPUBackend',
    'TPUBackend',
    'get_backend',
]
