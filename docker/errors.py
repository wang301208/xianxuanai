"""Minimal stub exceptions for the Docker SDK."""


class DockerException(Exception):
    """Base exception raised by the Docker stub."""


class NotFound(DockerException):
    """Raised when a requested container or resource cannot be found."""


class ImageNotFound(NotFound):
    """Raised when an image is not available locally."""
