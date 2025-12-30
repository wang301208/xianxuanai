"""AutoGPT 向量内存工具模块。

本模块提供了文本嵌入向量的生成和处理功能，支持多种输入格式和嵌入提供者。
是 AutoGPT 向量内存系统的核心工具模块。

主要功能:
    - 文本嵌入向量生成
    - 多种输入格式支持
    - 插件系统集成
    - 批量处理优化

支持的输入格式:
    - 单个文本字符串
    - 标记化文本序列
    - 文本列表（批量处理）
    - 标记序列列表

设计特点:
    - 类型安全的重载函数
    - 插件优先的处理策略
    - 灵活的嵌入提供者支持
    - 高效的批量处理
"""

import logging
from contextlib import suppress
from typing import Any, Sequence, overload

import numpy as np

from autogpt.config import Config
from autogpt.core.resource.model_providers import EmbeddingModelProvider

logger = logging.getLogger(__name__)

# 嵌入向量类型定义，支持多种数值格式
Embedding = list[float] | list[np.float32] | np.ndarray[Any, np.dtype[np.float32]]
"""嵌入向量类型，可以是浮点数列表或 NumPy 数组"""

# 标记化文本类型定义
TText = Sequence[int]
"""标记化文本类型，表示为整数序列"""


@overload
async def get_embedding(
    input: str | TText, config: Config, embedding_provider: EmbeddingModelProvider
) -> Embedding:
    ...


@overload
async def get_embedding(
    input: list[str] | list[TText],
    config: Config,
    embedding_provider: EmbeddingModelProvider,
) -> list[Embedding]:
    ...


async def get_embedding(
    input: str | TText | list[str] | list[TText],
    config: Config,
    embedding_provider: EmbeddingModelProvider,
) -> Embedding | list[Embedding]:
    """Get an embedding from the ada model.

    Args:
        input: Input text to get embeddings for, encoded as a string or array of tokens.
            Multiple inputs may be given as a list of strings or token arrays.
        embedding_provider: The provider to create embeddings.

    Returns:
        List[float]: The embedding.
    """
    multiple = isinstance(input, list) and all(not isinstance(i, int) for i in input)

    if isinstance(input, str):
        input = input.replace("\n", " ")

        with suppress(NotImplementedError):
            return _get_embedding_with_plugin(input, config)

    elif multiple and isinstance(input[0], str):
        input = [text.replace("\n", " ") for text in input]

        with suppress(NotImplementedError):
            return [_get_embedding_with_plugin(i, config) for i in input]

    model = config.embedding_model

    logger.debug(
        f"Getting embedding{f's for {len(input)} inputs' if multiple else ''}"
        f" with model '{model}'"
    )

    if not multiple:
        return (
            await embedding_provider.create_embedding(
                text=input,
                model_name=model,
                embedding_parser=lambda e: e,
            )
        ).embedding
    else:
        embeddings = []
        for text in input:
            result = await embedding_provider.create_embedding(
                text=text,
                model_name=model,
                embedding_parser=lambda e: e,
            )
            embeddings.append(result.embedding)
        return embeddings


def _get_embedding_with_plugin(text: str, config: Config) -> Embedding:
    for plugin in config.plugins:
        if plugin.can_handle_text_embedding(text):
            embedding = plugin.handle_text_embedding(text)
            if embedding is not None:
                return embedding

    raise NotImplementedError
