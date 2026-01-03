"""AutoGPT JSON 工具模块。

本模块提供了增强的 JSON 解析功能，能够处理各种不规范的 JSON 格式，
特别适用于处理 AI 模型生成的可能包含语法错误的 JSON 内容。

主要功能:
    - 容错的 JSON 解析
    - 从文本中提取 JSON 对象
    - 支持 Markdown 代码块格式
    - 处理各种 JSON 语法变体

技术特点:
    - 使用 demjson3 库提供强大的容错能力
    - 支持多种数字格式和编码
    - 自动处理注释和多余的空白字符
    - 提供详细的错误诊断信息

应用场景:
    - 解析 AI 模型的 JSON 响应
    - 处理用户输入的 JSON 数据
    - 从文档中提取结构化数据
    - 配置文件的宽松解析
"""

import logging
import re
import json
from typing import Any

try:  # optional dependency
    import demjson3  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency absent
    demjson3 = None  # type: ignore[assignment]

# 获取模块日志记录器
logger = logging.getLogger(__name__)


def json_loads(json_str: str) -> Any:
    """解析 JSON 字符串，容忍各种语法问题。
    
    这是一个增强版的 JSON 解析函数，能够处理标准 JSON 解析器
    无法处理的各种格式问题，特别适用于处理 AI 生成的内容。
    
    支持的语法变体:
        - 缺失、多余和尾随逗号
        - 字符串字面量外的多余换行和空白
        - 冒号和逗号后的不一致间距
        - 缺失的闭合括号或大括号
        - 各种数字格式：二进制、十六进制、八进制、小数点变体
        - 不同的字符编码
        - 包围的 Markdown 代码块
        - 注释内容
    
    参数:
        json_str: 要解析的 JSON 字符串
        
    返回:
        Any: 解析后的 JSON 对象，与内置 json.loads 相同
        
    异常:
        ValueError: 当 JSON 字符串无法解析时抛出
        
    使用示例:
        >>> json_loads('{"name": "test",}')  # 尾随逗号
        {'name': 'test'}
        >>> json_loads('```json\
{"key": "value"}\
```')  # Markdown 格式
        {'key': 'value'}
    """
    # Remove possible code block
    pattern = r"```(?:json|JSON)*([\s\S]*?)```"
    match = re.search(pattern, json_str)

    if match:
        json_str = match.group(1).strip()

    if demjson3 is None:
        return json.loads(json_str)

    json_result = demjson3.decode(json_str, return_errors=True)
    assert json_result is not None  # by virtue of return_errors=True

    if json_result.errors:
        logger.debug(
            "JSON parse errors:\n" + "\n".join(str(e) for e in json_result.errors)
        )

    if json_result.object in (demjson3.syntax_error, demjson3.undefined):
        raise ValueError(
            f"Failed to parse JSON string: {json_str}", *json_result.errors
        )

    return json_result.object


def extract_dict_from_json(json_str: str) -> dict[str, Any]:
    # Sometimes the response includes the JSON in a code block with ```
    pattern = r"```(?:json|JSON)*([\s\S]*?)```"
    match = re.search(pattern, json_str)

    if match:
        json_str = match.group(1).strip()
    else:
        # The string may contain JSON.
        json_pattern = r"{[\s\S]*}"
        match = re.search(json_pattern, json_str)

        if match:
            json_str = match.group()

    result = json_loads(json_str)
    if not isinstance(result, dict):
        raise ValueError(
            f"Response '''{json_str}''' evaluated to non-dict value {repr(result)}"
        )
    return result


def extract_list_from_json(json_str: str) -> list[Any]:
    # Sometimes the response includes the JSON in a code block with ```
    pattern = r"```(?:json|JSON)*([\s\S]*?)```"
    match = re.search(pattern, json_str)

    if match:
        json_str = match.group(1).strip()
    else:
        # The string may contain JSON.
        json_pattern = r"\[[\s\S]*\]"
        match = re.search(json_pattern, json_str)

        if match:
            json_str = match.group()

    result = json_loads(json_str)
    if not isinstance(result, list):
        raise ValueError(
            f"Response '''{json_str}''' evaluated to non-list value {repr(result)}"
        )
    return result
