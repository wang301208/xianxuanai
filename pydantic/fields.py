"""Minimal subset of Pydantic's :mod:`pydantic.fields` module used in tests.

This light-weight compatibility layer only implements the pieces of the
``pydantic`` API that are required by the unit tests that accompany this kata.
It is **not** a drop-in replacement for the real library, but it provides
enough structure for configuration models that rely on ``Field`` metadata.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict

__all__ = [
    "Undefined",
    "UndefinedType",
    "FieldInfo",
    "Field",
    "ModelField",
]


class UndefinedType:
    """Sentinel used to represent an undefined default value."""

    def __repr__(self) -> str:  # pragma: no cover - simple debug helper
        return "Undefined"


Undefined = UndefinedType()


class FieldInfo:
    """Container for metadata associated with a model field."""

    def __init__(
        self,
        default: Any = Undefined,
        *,
        default_factory: Callable[[], Any] | None = None,
        **extra: Any,
    ) -> None:
        self.default = default
        self.default_factory = default_factory
        self.extra: Dict[str, Any] = dict(extra)

    def copy(self) -> "FieldInfo":
        """Return a shallow copy of this ``FieldInfo`` instance."""

        return FieldInfo(
            default=self.default,
            default_factory=self.default_factory,
            **deepcopy(self.extra),
        )


def Field(
    default: Any = Undefined,
    *_,
    default_factory: Callable[[], Any] | None = None,
    **kwargs: Any,
) -> FieldInfo:
    """Return a :class:`FieldInfo` describing a model attribute.

    Only the arguments that are exercised in the tests are implemented. Extra
    keyword arguments are stored on the ``FieldInfo.extra`` mapping so calling
    code can inspect them just like it would with the real ``pydantic``.
    """

    return FieldInfo(default=default, default_factory=default_factory, **kwargs)


class ModelField:
    """Simplified stand-in for :class:`pydantic.fields.ModelField`."""

    def __init__(
        self,
        name: str,
        annotation: Any,
        field_info: FieldInfo,
        *,
        default: Any = Undefined,
        default_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.name = name
        self.type_ = annotation
        self.annotation = annotation
        self.outer_type_ = annotation
        self.field_info = field_info
        self.default = default
        self.default_factory = default_factory

    @property
    def required(self) -> bool:
        """Return ``True`` when the field requires an explicit value."""

        return (
            self.default in (Undefined, ...)
            and self.default_factory is None
        )

    def copy(self) -> "ModelField":
        return ModelField(
            self.name,
            self.annotation,
            self.field_info.copy(),
            default=self.default,
            default_factory=self.default_factory,
        )

