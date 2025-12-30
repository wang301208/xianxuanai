"""AutoGPT 项目级异常处理工具。

本模块提供了 AutoGPT 项目中统一的异常处理机制，包括自定义异常类
和标准化的错误处理工具函数。

主要功能:
    - 定义项目特定的异常基类
    - 提供标准化的异常日志记录
    - 统一的错误信息格式化

设计原则:
    - 异常层次结构清晰
    - 错误信息标准化
    - 便于调试和监控
    - 支持国际化和本地化
"""

from __future__ import annotations

import logging
from typing import Optional


class AutoGPTException(Exception):
    """AutoGPT 项目特定异常的基类。
    
    所有 AutoGPT 项目中的自定义异常都应该继承此类，
    以便于统一的异常处理和错误追踪。
    
    特点:
        - 提供项目级异常标识
        - 支持异常链和上下文传递
        - 便于异常分类和处理
        - 集成日志记录和监控
        
    使用示例:
        class ConfigurationError(AutoGPTException):
            '''配置错误异常'''
            pass
            
        class AgentExecutionError(AutoGPTException):
            '''代理执行错误异常'''
            pass
    """
    pass


def log_and_format_exception(
    exc: Exception, logger: Optional[logging.Logger] = None
) -> dict[str, str]:
    """记录异常日志并返回标准化的错误表示。
    
    这个函数提供了统一的异常处理流程，确保所有异常都被
    正确记录并以一致的格式返回给调用者。
    
    参数:
        exc: 要处理的异常对象
        logger: 可选的日志记录器，如果未提供则使用默认记录器
        
    返回:
        dict[str, str]: 包含错误类型和消息的标准化字典
            - error_type: 异常类的名称
            - message: 异常的字符串表示
            
    功能特点:
        - 自动记录完整的异常堆栈跟踪
        - 提供结构化的错误信息
        - 支持自定义日志记录器
        - 便于错误监控和分析
        
    使用示例:
        try:
            # 一些可能出错的操作
            risky_operation()
        except Exception as e:
            error_info = log_and_format_exception(e)
            return {"success": False, **error_info}
    """
    # 使用提供的日志记录器或获取当前模块的默认记录器
    log = logger or logging.getLogger(__name__)
    
    # 记录异常的完整信息，包括堆栈跟踪
    log.exception("%s: %s", type(exc).__name__, exc)
    
    # 返回标准化的错误信息字典
    return {
        "error_type": type(exc).__name__,  # 异常类型名称
        "message": str(exc)                # 异常消息内容
    }
