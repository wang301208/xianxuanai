"""Simplified implementation of Pydantic's ``BaseModel`` machinery."""

from __future__ import annotations

import abc
import json
from copy import deepcopy
from typing import Any, Dict, Iterable, Tuple, Type, TypeVar

from .decorators import validator  # re-exported for convenience
from .errors import ValidationError
from .fields import Field, FieldInfo, ModelField, Undefined, UndefinedType
from .types import SecretBytes, SecretField, SecretStr

__all__ = [
    "BaseModel",
    "BaseSettings",
    "ModelMetaclass",
    "ValidationError",
    "Field",
    "SecretStr",
    "SecretBytes",
    "SecretField",
    "validator",
]

T = TypeVar("T", bound="BaseModel")


def _deep_copy(value: Any) -> Any:
    try:
        return deepcopy(value)
    except Exception:  # pragma: no cover - defensive guard
        return value


def _to_dict(value: Any, *, exclude_none: bool) -> Any:
    if isinstance(value, BaseModel):
        return value.dict(exclude_none=exclude_none)
    if isinstance(value, list):
        return [_to_dict(v, exclude_none=exclude_none) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_dict(v, exclude_none=exclude_none) for v in value)
    if isinstance(value, dict):
        return {
            k: _to_dict(v, exclude_none=exclude_none)
            for k, v in value.items()
            if not (exclude_none and v is None)
        }
    return value


class ModelMetaclass(abc.ABCMeta):
    """Metaclass that builds the ``__fields__`` mapping for ``BaseModel``."""

    def __new__(
        mcls,
        name: str,
        bases: Tuple[type, ...],
        namespace: Dict[str, Any],
        **kwargs: Any,
    ) -> "ModelMetaclass":
        annotations: Dict[str, Any] = {}
        fields: Dict[str, ModelField] = {}

        for base in reversed(bases):
            annotations.update(getattr(base, "__annotations__", {}))
            if hasattr(base, "__fields__"):
                fields.update({k: v.copy() for k, v in base.__fields__.items()})

        own_annotations = namespace.get("__annotations__", {})
        annotations.update(own_annotations)

        namespace = dict(namespace)
        for field_name, annotation in own_annotations.items():
            raw_default = namespace.get(field_name, Undefined)
            if isinstance(raw_default, FieldInfo):
                field_info = raw_default
                default = raw_default.default
                default_factory = raw_default.default_factory
            elif raw_default is Undefined:
                field_info = FieldInfo(default=Undefined)
                default = Undefined
                default_factory = None
            else:
                field_info = FieldInfo(default=raw_default)
                default = raw_default
                default_factory = None

            fields[field_name] = ModelField(
                field_name,
                annotation,
                field_info,
                default=default,
                default_factory=default_factory,
            )

            if field_name in namespace:
                del namespace[field_name]

        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        cls.__annotations__ = annotations
        cls.__fields__ = fields
        return cls


class BaseModel(metaclass=ModelMetaclass):
    """Very small subset of ``pydantic.BaseModel`` sufficient for the tests."""

    Config = type("Config", (), {})

    def __init__(self, **data: Any) -> None:
        errors = []
        values: Dict[str, Any] = {}
        remaining = dict(data)

        for name, field in self.__class__.__fields__.items():
            if name in remaining:
                value = remaining.pop(name)
            else:
                if field.default not in (Undefined, ...):
                    value = _deep_copy(field.default)
                elif field.default_factory is not None:
                    value = field.default_factory()
                else:
                    errors.append(
                        {
                            "loc": (name,),
                            "msg": "field required",
                            "type": "value_error.missing",
                        }
                    )
                    continue
            values[name] = value

        if errors:
            raise ValidationError(errors)

        for name, value in values.items():
            setattr(self, name, value)

        for extra_name, extra_value in remaining.items():
            setattr(self, extra_name, extra_value)

    def dict(self, *, exclude_none: bool = False) -> Dict[str, Any]:
        return {
            name: _to_dict(getattr(self, name), exclude_none=exclude_none)
            for name in self.__class__.__fields__
            if not (exclude_none and getattr(self, name) is None)
        }

    def model_dump(self, *, exclude_none: bool = False) -> Dict[str, Any]:
        return self.dict(exclude_none=exclude_none)

    def copy(self: T, *, update: Dict[str, Any] | None = None) -> T:
        data = self.dict()
        if update:
            data.update(update)
        return self.__class__(**data)

    def model_copy(self: T, *, update: Dict[str, Any] | None = None) -> T:
        return self.copy(update=update)

    def json(self, *, encoder=None, exclude_none: bool = False, **kwargs: Any) -> str:
        def default(value: Any) -> Any:
            if encoder is not None:
                try:
                    return encoder(value)
                except TypeError:
                    pass
            if isinstance(value, BaseModel):
                return value.dict(exclude_none=exclude_none)
            if hasattr(value, "get_secret_value"):
                return value.get_secret_value()
            raise TypeError(f"Object of type {type(value)!r} is not JSON serialisable")

        return json.dumps(
            self.dict(exclude_none=exclude_none),
            default=default,
            **kwargs,
        )

    def model_dump_json(self, *, exclude_none: bool = False, **kwargs: Any) -> str:
        return self.json(exclude_none=exclude_none, **kwargs)

    @classmethod
    def parse_obj(cls: Type[T], obj: Any) -> T:
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls(**obj)
        raise TypeError(f"Object of type {type(obj)!r} is not supported")

    @classmethod
    def update_forward_refs(cls, **_localns: Any) -> None:
        # No-op: the compatibility layer does not need to resolve forward refs.
        return None

    def __iter__(self) -> Iterable[Tuple[str, Any]]:
        for name in self.__class__.__fields__:
            yield name, getattr(self, name)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        field_str = ", ".join(f"{k}={getattr(self, k)!r}" for k in self.__class__.__fields__)
        return f"{self.__class__.__name__}({field_str})"

    @classmethod
    def __class_getitem__(cls, _params: Any) -> Type["BaseModel"]:
        # Generic parameters are ignored by this lightweight implementation.
        return cls


class BaseSettings(BaseModel):
    """Alias for ``BaseModel`` used in a few configuration helpers."""

    pass

