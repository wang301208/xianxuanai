"""AutoGPT 系统配置架构定义。

本模块定义了 AutoGPT 系统的配置架构，包括用户可配置字段、
系统配置基类和配置处理工具函数。提供了灵活的配置管理机制，
支持从环境变量、配置文件等多种来源加载配置。

主要组件:
    - UserConfigurable: 用户可配置字段装饰器
    - SystemConfiguration: 系统配置基类
    - SystemSettings: 系统设置基类
    - Configurable: 可配置对象基类

设计特点:
    - 基于 Pydantic 的类型安全配置
    - 支持环境变量自动映射
    - 递归配置处理
    - 配置验证和错误处理
"""

import abc
import os
import typing
from typing import Any, Callable, Generic, Optional, Type, TypeVar, get_args

from pydantic import BaseModel, Field, ValidationError
from pydantic.fields import ModelField, Undefined, UndefinedType
from pydantic.main import ModelMetaclass

# 泛型类型变量定义
T = TypeVar("T")  # 通用类型变量
M = TypeVar("M", bound=BaseModel)  # 绑定到 BaseModel 的类型变量


def UserConfigurable(
    default: T | UndefinedType = Undefined,
    *args,
    default_factory: Optional[Callable[[], T]] = None,
    from_env: Optional[str | Callable[[], T | None]] = None,
    description: str = "",
    **kwargs,
) -> T:
    """创建用户可配置的字段。
    
    这是一个装饰器函数，用于标记字段为用户可配置，
    支持从环境变量、默认值或工厂函数获取配置值。
    
    参数:
        default: 字段的默认值
        *args: 传递给 Field 的位置参数
        default_factory: 默认值工厂函数
        from_env: 环境变量名或获取函数
        description: 字段描述信息
        **kwargs: 传递给 Field 的关键字参数
        
    返回:
        T: 配置字段值
        
    注意:
        标记为 user_configurable=True 的字段会被配置系统特殊处理，
        可以通过环境变量、配置文件等方式进行配置。
        
    TODO: 
        使用此信息自动生成应用程序配置文档
    """
    return Field(
        default,
        *args,
        default_factory=default_factory,
        from_env=from_env,
        description=description,
        **kwargs,
        user_configurable=True,  # 标记为用户可配置字段
    )


class SystemConfiguration(BaseModel):
    def get_user_config(self) -> dict[str, Any]:
        return _recurse_user_config_values(self)

    @classmethod
    def from_env(cls):
        """
        Initializes the config object from environment variables.

        Environment variables are mapped to UserConfigurable fields using the from_env
        attribute that can be passed to UserConfigurable.
        """

        def infer_field_value(field: ModelField):
            field_info = field.field_info
            default_value = (
                field.default
                if field.default not in (None, Undefined)
                else (field.default_factory() if field.default_factory else Undefined)
            )
            if from_env := field_info.extra.get("from_env"):
                val_from_env = (
                    os.getenv(from_env) if type(from_env) is str else from_env()
                )
                if val_from_env is not None:
                    return val_from_env
            return default_value

        return _recursive_init_model(cls, infer_field_value)

    class Config:
        extra = "forbid"
        use_enum_values = True
        validate_assignment = True


SC = TypeVar("SC", bound=SystemConfiguration)


class SystemSettings(BaseModel):
    """A base class for all system settings."""

    name: str = ""
    description: str = ""

    class Config:
        extra = "forbid"
        use_enum_values = True
        validate_assignment = True


S = TypeVar("S", bound=SystemSettings)


class Configurable(abc.ABC, Generic[S]):
    """A base class for all configurable objects."""

    prefix: str = ""
    default_settings: typing.ClassVar[S]

    @classmethod
    def get_user_config(cls) -> dict[str, Any]:
        return _recurse_user_config_values(cls.default_settings)

    @classmethod
    def build_agent_configuration(cls, overrides: dict = {}) -> S:
        """Process the configuration for this object."""

        base_config = _update_user_config_from_env(cls.default_settings)
        final_configuration = deep_update(base_config, overrides)

        return cls.default_settings.__class__.parse_obj(final_configuration)


def _update_user_config_from_env(instance: BaseModel) -> dict[str, Any]:
    """
    Update config fields of a Pydantic model instance from environment variables.

    Precedence:
    1. Non-default value already on the instance
    2. Value returned by `from_env()`
    3. Default value for the field

    Params:
        instance: The Pydantic model instance.

    Returns:
        The user config fields of the instance.
    """

    def infer_field_value(field: ModelField, value):
        field_info = field.field_info
        default_value = (
            field.default
            if field.default not in (None, Undefined)
            else (field.default_factory() if field.default_factory else None)
        )
        if value == default_value and (from_env := field_info.extra.get("from_env")):
            val_from_env = os.getenv(from_env) if type(from_env) is str else from_env()
            if val_from_env is not None:
                return val_from_env
        return value

    def init_sub_config(model: Type[SC]) -> SC | None:
        try:
            return model.from_env()
        except ValidationError as e:
            # Gracefully handle missing fields
            if all(e["type"] == "value_error.missing" for e in e.errors()):
                return None
            raise

    return _recurse_user_config_fields(instance, infer_field_value, init_sub_config)


def _recursive_init_model(
    model: Type[M],
    infer_field_value: Callable[[ModelField], Any],
) -> M:
    """
    Recursively initialize the user configuration fields of a Pydantic model.

    Parameters:
        model: The Pydantic model type.
        infer_field_value: A callback function to infer the value of each field.
            Parameters:
                ModelField: The Pydantic ModelField object describing the field.

    Returns:
        BaseModel: An instance of the model with the initialized configuration.
    """
    user_config_fields = {}
    for name, field in model.__fields__.items():
        if "user_configurable" in field.field_info.extra:
            user_config_fields[name] = infer_field_value(field)
        elif type(field.outer_type_) is ModelMetaclass and issubclass(
            field.outer_type_, SystemConfiguration
        ):
            try:
                user_config_fields[name] = _recursive_init_model(
                    model=field.outer_type_,
                    infer_field_value=infer_field_value,
                )
            except ValidationError as e:
                # Gracefully handle missing fields
                if all(e["type"] == "value_error.missing" for e in e.errors()):
                    user_config_fields[name] = None
                raise

    user_config_fields = remove_none_items(user_config_fields)

    return model.parse_obj(user_config_fields)


def _recurse_user_config_fields(
    model: BaseModel,
    infer_field_value: Callable[[ModelField, Any], Any],
    init_sub_config: Optional[
        Callable[[Type[SystemConfiguration]], SystemConfiguration | None]
    ] = None,
) -> dict[str, Any]:
    """
    Recursively process the user configuration fields of a Pydantic model instance.

    Params:
        model: The Pydantic model to iterate over.
        infer_field_value: A callback function to process each field.
            Params:
                ModelField: The Pydantic ModelField object describing the field.
                Any: The current value of the field.
        init_sub_config: An optional callback function to initialize a sub-config.
            Params:
                Type[SystemConfiguration]: The type of the sub-config to initialize.

    Returns:
        dict[str, Any]: The processed user configuration fields of the instance.
    """
    user_config_fields = {}

    for name, field in model.__fields__.items():
        value = getattr(model, name)

        # Handle individual field
        if "user_configurable" in field.field_info.extra:
            user_config_fields[name] = infer_field_value(field, value)

        # Recurse into nested config object
        elif isinstance(value, SystemConfiguration):
            user_config_fields[name] = _recurse_user_config_fields(
                model=value,
                infer_field_value=infer_field_value,
                init_sub_config=init_sub_config,
            )

        # Recurse into optional nested config object
        elif value is None and init_sub_config:
            args = ()
            try:
                args = get_args(field.annotation)
            except Exception:
                args = ()
            if not args:
                try:
                    args = get_args(field.outer_type_)
                except Exception:
                    args = ()
            candidates = [t for t in args if t is not type(None)]  # noqa: E721
            field_type = candidates[0] if len(candidates) == 1 else None
            if field_type is not None and type(field_type) is ModelMetaclass and issubclass(
                field_type, SystemConfiguration
            ):
                sub_config = init_sub_config(field_type)
                if sub_config:
                    user_config_fields[name] = _recurse_user_config_fields(
                        model=sub_config,
                        infer_field_value=infer_field_value,
                        init_sub_config=init_sub_config,
                    )

        elif isinstance(value, list) and all(
            isinstance(i, SystemConfiguration) for i in value
        ):
            user_config_fields[name] = [
                _recurse_user_config_fields(i, infer_field_value, init_sub_config)
                for i in value
            ]
        elif isinstance(value, dict) and all(
            isinstance(i, SystemConfiguration) for i in value.values()
        ):
            user_config_fields[name] = {
                k: _recurse_user_config_fields(v, infer_field_value, init_sub_config)
                for k, v in value.items()
            }

    return user_config_fields


def _recurse_user_config_values(
    instance: BaseModel,
    get_field_value: Callable[[ModelField, T], T] = lambda _, v: v,
) -> dict[str, Any]:
    """
    This function recursively traverses the user configuration values in a Pydantic
    model instance.

    Params:
        instance: A Pydantic model instance.
        get_field_value: A callback function to process each field. Parameters:
            ModelField: The Pydantic ModelField object that describes the field.
            Any: The current value of the field.

    Returns:
        A dictionary containing the processed user configuration fields of the instance.
    """
    user_config_values = {}

    for name, value in instance.__dict__.items():
        field = instance.__fields__[name]
        if "user_configurable" in field.field_info.extra:
            user_config_values[name] = get_field_value(field, value)
        elif isinstance(value, SystemConfiguration):
            user_config_values[name] = _recurse_user_config_values(
                instance=value, get_field_value=get_field_value
            )
        elif isinstance(value, list) and all(
            isinstance(i, SystemConfiguration) for i in value
        ):
            user_config_values[name] = [
                _recurse_user_config_values(i, get_field_value) for i in value
            ]
        elif isinstance(value, dict) and all(
            isinstance(i, SystemConfiguration) for i in value.values()
        ):
            user_config_values[name] = {
                k: _recurse_user_config_values(v, get_field_value)
                for k, v in value.items()
            }

    return user_config_values


def _get_non_default_user_config_values(instance: BaseModel) -> dict[str, Any]:
    """
    Get the non-default user config fields of a Pydantic model instance.

    Params:
        instance: The Pydantic model instance.

    Returns:
        dict[str, Any]: The non-default user config values on the instance.
    """

    def get_field_value(field: ModelField, value):
        default = field.default_factory() if field.default_factory else field.default
        if value != default:
            return value

    return remove_none_items(_recurse_user_config_values(instance, get_field_value))


def deep_update(original_dict: dict, update_dict: dict) -> dict:
    """
    Recursively update a dictionary.

    Params:
        original_dict (dict): The dictionary to be updated.
        update_dict (dict): The dictionary to update with.

    Returns:
        dict: The updated dictionary.
    """
    for key, value in update_dict.items():
        if (
            key in original_dict
            and isinstance(original_dict[key], dict)
            and isinstance(value, dict)
        ):
            original_dict[key] = deep_update(original_dict[key], value)
        else:
            original_dict[key] = value
    return original_dict


def remove_none_items(d):
    if isinstance(d, dict):
        return {
            k: remove_none_items(v) for k, v in d.items() if v not in (None, Undefined)
        }
    return d
