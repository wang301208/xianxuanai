"""Simplified implementation of Pydantic's ``BaseModel`` machinery."""

from __future__ import annotations

import abc
import json
from copy import deepcopy
import sys
import types
from typing import Any, Dict, Iterable, Tuple, Type, TypeVar, Union, get_args, get_origin, get_type_hints

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


def _is_basemodel_type(annotation: Any) -> bool:
    try:
        return isinstance(annotation, type) and issubclass(annotation, BaseModel)
    except Exception:
        return False


def _coerce_value(annotation: Any, value: Any) -> Any:
    if value is None:
        return None

    if _is_basemodel_type(annotation):
        if isinstance(value, annotation):
            return value
        if isinstance(value, dict):
            return annotation.parse_obj(value)
        return value

    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        args = [arg for arg in get_args(annotation) if arg is not type(None)]  # noqa: E721
        if len(args) == 1:
            return _coerce_value(args[0], value)
        return value

    if origin is list and isinstance(value, list):
        args = get_args(annotation)
        item_type = args[0] if args else None
        if item_type and _is_basemodel_type(item_type):
            return [item_type.parse_obj(v) if isinstance(v, dict) else v for v in value]
        return value

    if origin is tuple and isinstance(value, tuple):
        args = get_args(annotation)
        item_type = args[0] if args else None
        if item_type and _is_basemodel_type(item_type):
            return tuple(item_type.parse_obj(v) if isinstance(v, dict) else v for v in value)
        return value

    if origin is dict and isinstance(value, dict):
        args = get_args(annotation)
        value_type = args[1] if len(args) >= 2 else None
        if value_type and _is_basemodel_type(value_type):
            return {k: value_type.parse_obj(v) if isinstance(v, dict) else v for k, v in value.items()}
        return value

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

        try:
            module = sys.modules.get(cls.__module__)
            globalns = getattr(module, "__dict__", {}) if module is not None else {}
            resolved = get_type_hints(cls, globalns=globalns, localns=dict(vars(cls)))
        except Exception:
            resolved = {}

        if resolved:
            for field_name, field in cls.__fields__.items():
                resolved_type = resolved.get(field_name)
                if resolved_type is None:
                    continue
                field.annotation = resolved_type
                field.outer_type_ = resolved_type
                field.type_ = resolved_type
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
            values[name] = _coerce_value(field.annotation, value)

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

    def copy(
        self: T,
        *,
        deep: bool = False,
        update: Dict[str, Any] | None = None,
    ) -> T:
        """Return a copy of this model.

        This compatibility layer supports the common ``copy(deep=True)`` usage
        found throughout the upstream AutoGPT codebase.
        """

        if deep:
            cloned: T = _deep_copy(self)
        else:
            cloned = self.__class__(**self.dict())

        if update:
            for key, value in update.items():
                setattr(cloned, key, value)
        return cloned

    def model_copy(
        self: T,
        *,
        deep: bool = False,
        update: Dict[str, Any] | None = None,
    ) -> T:
        return self.copy(deep=deep, update=update)

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
