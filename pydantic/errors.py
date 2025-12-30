"""Minimal error types mimicking Pydantic's interface."""

from __future__ import annotations

from typing import Any, Iterable, List

__all__ = ["ValidationError"]


class ValidationError(Exception):
    """Simple ``ValidationError`` carrying structured error details."""

    def __init__(self, errors: Iterable[dict[str, Any]]) -> None:
        self._errors: List[dict[str, Any]] = list(errors)
        message = ", ".join(err.get("msg", str(err)) for err in self._errors)
        super().__init__(message or "Validation error")

    def errors(self) -> List[dict[str, Any]]:
        return list(self._errors)

