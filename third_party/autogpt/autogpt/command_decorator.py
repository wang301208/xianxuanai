"""AutoGPT 命令装饰器模块。

本模块提供了用于创建 AutoGPT 命令的装饰器功能，将普通函数转换为
系统可识别和执行的命令对象。命令装饰器是 AutoGPT 能力系统的核心组件。

主要功能:
    - 将函数转换为命令对象
    - 提供参数验证和类型检查
    - 支持命令的启用/禁用控制
    - 管理命令别名和可用性

设计优势:
    - 声明式的命令定义方式
    - 类型安全的参数处理
    - 灵活的条件控制机制
    - 统一的命令接口规范
"""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, ParamSpec, TypeVar

if TYPE_CHECKING:
    from autogpt.agents.base import BaseAgent
    from autogpt.config import Config

from autogpt.core.utils.json_schema import JSONSchema
from autogpt.models.command import Command, CommandOutput, CommandParameter

# AutoGPT 命令的唯一标识符
# 用于在运行时识别哪些函数是 AutoGPT 命令
AUTO_GPT_COMMAND_IDENTIFIER = "auto_gpt_command"

# 泛型类型定义
P = ParamSpec("P")  # 参数规范类型变量
CO = TypeVar("CO", bound=CommandOutput)  # 命令输出类型变量


def command(
    name: str,
    description: str,
    parameters: dict[str, JSONSchema],
    enabled: Literal[True] | Callable[[Config], bool] = True,
    disabled_reason: Optional[str] = None,
    aliases: list[str] = [],
    available: bool | Callable[[BaseAgent], bool] = True,
) -> Callable[[Callable[P, CO]], Callable[P, CO]]:
    """命令装饰器，用于将普通函数转换为 AutoGPT 命令对象。
    
    这个装饰器是 AutoGPT 命令系统的核心，它将普通的 Python 函数
    包装成系统可以识别、验证和执行的命令对象。
    
    参数:
        name: 命令名称，用于在系统中标识命令
        description: 命令描述，说明命令的功能和用途
        parameters: 命令参数定义，使用 JSONSchema 进行类型约束
        enabled: 命令是否启用，可以是布尔值或返回布尔值的函数
        disabled_reason: 命令被禁用时的原因说明
        aliases: 命令别名列表，提供命令的替代名称
        available: 命令是否可用，可以是布尔值或检查函数
        
    返回:
        装饰器函数，用于包装目标函数
        
    使用示例:
        @command(
            name="read_file",
            description="读取文件内容",
            parameters={
                "filename": JSONSchema(type="string", description="文件名")
            }
        )
        def read_file(filename: str) -> str:
            # 实现文件读取逻辑
            pass
            
    设计特点:
        - 支持同步和异步函数
        - 保持原函数的类型签名
        - 提供灵活的启用/禁用控制
        - 集成参数验证机制
    """

    def decorator(func: Callable[P, CO]) -> Callable[P, CO]:
        typed_parameters = [
            CommandParameter(
                name=param_name,
                spec=spec,
            )
            for param_name, spec in parameters.items()
        ]
        cmd = Command(
            name=name,
            description=description,
            method=func,
            parameters=typed_parameters,
            enabled=enabled,
            disabled_reason=disabled_reason,
            aliases=aliases,
            available=available,
        )

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                return await func(*args, **kwargs)

        else:

            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                return func(*args, **kwargs)

        setattr(wrapper, "command", cmd)
        setattr(wrapper, AUTO_GPT_COMMAND_IDENTIFIER, True)

        return wrapper

    return decorator
