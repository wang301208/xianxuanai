"""AutoGPT 代理管理器模块。

本模块提供了 AutoGPT 代理的管理功能，包括代理的创建、
配置、生命周期管理等核心功能。

导出组件:
    - AgentManager: 代理管理器主类

主要功能:
    - 代理实例管理
    - 代理配置管理
    - 代理生命周期控制
    - 代理状态监控
"""

from .agent_manager import AgentManager

# 模块公开接口
__all__ = ["AgentManager"]
