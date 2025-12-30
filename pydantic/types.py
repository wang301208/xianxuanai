"""Minimal secret field implementations used in tests."""

from __future__ import annotations

from typing import Any

__all__ = ["SecretStr", "SecretBytes", "SecretField"]


class _SecretBase:
    """Common functionality for simple secret value containers."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def get_secret_value(self) -> Any:
        return self._value

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"{self.__class__.__name__}('**********')"

    def __str__(self) -> str:
        return "**********"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self._value == other._value
        return self._value == other


class SecretStr(_SecretBase):
    def __init__(self, value: str | None) -> None:
        super().__init__(None if value is None else str(value))


class SecretBytes(_SecretBase):
    def __init__(self, value: bytes | bytearray | None) -> None:
        if value is None:
            secret = None
        elif isinstance(value, (bytes, bytearray)):
            secret = bytes(value)
        else:
            secret = bytes(str(value), "utf-8")
        super().__init__(secret)


class SecretField(_SecretBase):
    """Generic secret container used in a handful of locations."""

    pass

