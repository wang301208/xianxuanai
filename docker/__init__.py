"""Minimal stub of the Docker SDK used for unit testing."""

from __future__ import annotations

from .errors import DockerException, ImageNotFound, NotFound
from .models import Container

__all__ = [
    "DockerException",
    "ImageNotFound",
    "NotFound",
    "APIClient",
    "from_env",
]


class _Containers:
    def get(self, name: str):  # pragma: no cover - stubbed behaviour
        raise NotFound(f"Container '{name}' not found")

    def run(self, *args, **kwargs):  # pragma: no cover - stubbed behaviour
        raise DockerException("Docker is not available in the test environment")


class _Images:
    def get(self, name: str):  # pragma: no cover - stubbed behaviour
        raise ImageNotFound(f"Image '{name}' not found")


class _DockerClient:
    def __init__(self) -> None:
        self.containers = _Containers()
        self.images = _Images()

    def info(self) -> dict:
        raise DockerException("Docker is not available")


class APIClient:
    def pull(self, *args, **kwargs):  # pragma: no cover - stubbed behaviour
        return iter(())


def from_env() -> _DockerClient:  # pragma: no cover - deterministic stub
    return _DockerClient()
