"""AutoGPT 通用工具模块。

本模块提供了 AutoGPT 项目中各个组件共享的通用工具和实用程序，
包括异常处理、异步工具、概念建模和情感分析等核心功能。

主要组件:
    - 异常处理: 项目级异常类和错误格式化工具
    - 异步工具: 异步编程辅助函数
    - 概念建模: 知识图谱和概念关系表示
    - 情感分析: 情感状态检测和响应风格调整

设计目标:
    - 提供统一的基础设施组件
    - 减少代码重复和提高复用性
    - 确保项目范围内的一致性
    - 支持模块化和可扩展的架构
"""

from .exceptions import AutoGPTException, log_and_format_exception
from .async_utils import run_async
from .concepts import CausalRelation, ConceptNode, ConceptRelation
from .emotion import EmotionAnalyzer, EmotionState, adjust_response_style

# 公开的 API 接口列表
# 定义了模块对外暴露的所有公共接口
__all__ = [
    # 异常处理相关
    "AutoGPTException",           # AutoGPT 项目基础异常类
    "log_and_format_exception",   # 异常日志记录和格式化工具
    
    # 异步编程工具
    "run_async",                  # 异步协程执行工具
    
    # 概念建模组件
    "ConceptNode",                # 概念节点表示
    "ConceptRelation",            # 概念关系表示
    "CausalRelation",             # 因果关系表示
    
    # 情感分析组件
    "EmotionAnalyzer",            # 情感分析器
    "EmotionState",               # 情感状态表示
    "adjust_response_style",      # 响应风格调整函数
]
