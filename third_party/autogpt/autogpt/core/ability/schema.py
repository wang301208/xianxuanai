"""AutoGPT 能力系统的数据模式定义。

本模块定义了 AutoGPT 能力系统中使用的核心数据结构，
包括内容类型、知识表示和能力执行结果等。

主要组件:
    - ContentType: 内容类型枚举
    - Knowledge: 知识表示模型
    - AbilityResult: 能力执行结果模型

设计目标:
    - 标准化能力系统的数据交换格式
    - 提供类型安全的数据验证
    - 支持知识的结构化存储和检索
"""

import enum
from typing import Any

from pydantic import BaseModel


class ContentType(str, enum.Enum):
    """内容类型枚举。
    
    定义了系统中支持的不同内容类型，用于标识和处理
    不同格式的数据内容。
    
    支持的类型:
        TEXT: 纯文本内容
        CODE: 代码内容
        
    注意:
        这些类型的具体定义和处理方式待进一步确定。
    """
    TEXT = "text"  # 纯文本内容
    CODE = "code"  # 代码内容


class Knowledge(BaseModel):
    """知识表示模型。
    
    用于存储和传递系统中的知识信息，包括内容本身、
    内容类型和相关的元数据。
    
    属性:
        content: 知识的具体内容
        content_type: 内容类型（文本或代码等）
        content_metadata: 内容的元数据信息
        
    应用场景:
        - 存储从能力执行中获得的新知识
        - 在不同组件间传递结构化信息
        - 支持知识的分类和检索
    """
    content: str  # 知识内容
    content_type: ContentType  # 内容类型
    content_metadata: dict[str, Any]  # 内容元数据


class AbilityResult(BaseModel):
    """能力执行结果的标准响应结构。
    
    这是所有能力执行后返回的标准数据结构，
    包含执行状态、结果信息和可能产生的新知识。
    
    属性:
        ability_name: 执行的能力名称
        ability_args: 能力执行时使用的参数
        success: 执行是否成功
        message: 执行结果消息
        new_knowledge: 执行过程中产生的新知识（可选）
        
    设计优势:
        - 统一的结果格式便于处理和调试
        - 包含足够的信息用于日志记录和错误诊断
        - 支持知识的自动收集和存储
    """

    ability_name: str  # 能力名称
    ability_args: dict[str, str]  # 能力参数
    success: bool  # 执行成功标志
    message: str  # 结果消息
    new_knowledge: Knowledge = None  # 新产生的知识（可选）

    def summary(self) -> str:
        """生成能力执行结果的摘要字符串。
        
        返回:
            str: 包含能力名称、参数和结果消息的摘要
            
        格式:
            ability_name(arg1=value1, arg2=value2): message
            
        用途:
            - 日志记录
            - 调试输出
            - 用户界面显示
        """
        # 格式化参数列表
        kwargs = ", ".join(f"{k}={v}" for k, v in self.ability_args.items())
        return f"{self.ability_name}({kwargs}): {self.message}"
