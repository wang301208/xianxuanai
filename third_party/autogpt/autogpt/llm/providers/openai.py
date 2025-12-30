"""AutoGPT OpenAI 语言模型提供者模块。

本模块提供了与 OpenAI API 集成的功能，主要负责将 AutoGPT 的命令规范
转换为 OpenAI 函数调用格式，支持 GPT 模型的函数调用功能。

主要功能:
    - 命令规范转换
    - OpenAI 函数调用格式适配
    - 参数规范映射
    - 类型安全的函数定义

设计特点:
    - 遵循 OpenAI 函数调用规范
    - 支持动态命令集合
    - 提供类型安全的转换
    - 简洁的接口设计

参考文档:
    https://platform.openai.com/docs/guides/gpt/function-calling
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable, TypeVar

from autogpt.core.resource.model_providers import CompletionModelFunction
from autogpt.models.command import Command

logger = logging.getLogger(__name__)

# 泛型类型变量，用于类型安全的函数绑定
T = TypeVar("T", bound=Callable)


def get_openai_command_specs(
    commands: Iterable[Command],
) -> list[CompletionModelFunction]:
    """将代理的可用命令转换为 OpenAI 可消费的函数规范。

    该函数将 AutoGPT 的内部命令格式转换为符合 OpenAI 函数调用
    规范的格式，使 GPT 模型能够理解和调用这些命令。

    Args:
        commands: 可迭代的命令集合，包含代理的所有可用命令

    Returns:
        list[CompletionModelFunction]: OpenAI 函数规范列表，每个元素包含：
            - name: 函数名称
            - description: 函数描述
            - parameters: 参数规范字典

    转换过程:
        1. 提取命令名称和描述
        2. 转换参数规范格式
        3. 构建 OpenAI 兼容的函数定义
        4. 返回完整的函数规范列表

    使用场景:
        - GPT 模型函数调用
        - 命令能力暴露
        - API 规范转换
        - 动态函数注册

    参考:
        OpenAI 函数调用文档：
        https://platform.openai.com/docs/guides/gpt/function-calling
    """
    return [
        CompletionModelFunction(
            name=command.name,  # 命令名称
            description=command.description,  # 命令描述
            parameters={param.name: param.spec for param in command.parameters},  # 参数规范映射
        )
        for command in commands
    ]
