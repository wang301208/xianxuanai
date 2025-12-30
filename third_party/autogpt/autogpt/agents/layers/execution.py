from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from autogpt.core.ability import AbilityRegistry, SimpleAbilityRegistry
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.agent.layered import LayeredAgent


class ExecutionAgent(LayeredAgent):
    """负责从执行计划中执行能力的智能体层。
    
    该层在AutoGPT架构中充当执行引擎，接收结构化计划并通过
    能力注册系统执行指定的能力。它充当高级规划和低级能力执行之间的桥梁。
    
    主要职责：
    - 解析执行计划以提取能力名称和参数
    - 通过注册的能力注册表执行能力
    - 处理执行错误并将其路由到适当的层
    - 将成功的结果转发到后续处理层
    
    ExecutionAgent与分层架构集成，提供：
    - 集中化的能力执行
    - 错误处理和恢复机制
    - 结果转发和处理管道
    
    属性:
        _ability_registry: 包含该智能体可执行的所有可用能力的注册表
    """
=======

    def __init__(
        self,
        ability_registry: AbilityRegistry | SimpleAbilityRegistry,
        next_layer: Optional[LayeredAgent] = None,
    ) -> None:
        """使用能力注册表初始化执行智能体。
        
        参数:
            ability_registry: 包含该智能体可执行能力的注册表。
                             可以是完整的AbilityRegistry或SimpleAbilityRegistry。
            next_layer: 处理链中用于结果转发的可选下一层
        """
=======
        super().__init__(next_layer=next_layer)
        self._ability_registry = ability_registry

    @classmethod
    def from_workspace(
        cls, workspace_path: Path, logger: Any
    ) -> "ExecutionAgent":  # pragma: no cover - simple passthrough
        """从工作空间配置创建ExecutionAgent。
        
        注意：此方法当前未实现，因为ExecutionAgent
        需要显式的能力注册表配置。
        
        参数:
            workspace_path: 工作空间目录的路径
            logger: 智能体的日志记录器实例
            
        异常:
            NotImplementedError: 不支持此创建方法
        """
=======
        raise NotImplementedError("ExecutionAgent does not support workspace loading")

    def __repr__(self) -> str:  # pragma: no cover - simple representation
        """返回ExecutionAgent的字符串表示。
        
        返回:
            str: 显示能力注册表的字符串表示
        """
=======
        return f"ExecutionAgent(registry={self._ability_registry})"

    async def route_task(
        self, plan: dict[str, Any], *args: Any, **kwargs: Any
    ) -> AbilityResult:
        """执行执行计划中指定的能力。
        
        该方法通过以下步骤处理执行计划：
        1. 从计划中提取能力名称和参数
        2. 通过注册表执行能力
        3. 通过转发到下一层来处理任何执行错误
        4. 如果存在下一层，则将成功结果转发到下一层
        
        参数:
            plan: 包含以下内容的执行计划字典：
                 - 'next_ability': 要执行的能力名称
                 - 'ability_arguments': 传递给能力的参数（可选）
            *args: 任务处理的额外位置参数
            **kwargs: 任务处理的额外关键字参数
            
        返回:
            AbilityResult: 能力执行的结果，可能由后续层处理
                          
        异常:
            Exception: 如果没有下一层可用来处理执行错误，则重新抛出执行错误
        """
=======
        ability_name = plan.get("next_ability")
        ability_args = plan.get("ability_arguments", {})
        
        try:
            # Execute the specified ability with provided arguments
            result = await self._ability_registry.perform(ability_name, **ability_args)
        except Exception as err:
            # If execution fails and there's a next layer, forward the error
            if self.next_layer is not None:
                return await self.next_layer.route_task(err, *args, **kwargs)
            # Otherwise, re-raise the exception
            raise

        # Forward successful results to the next layer if present
        if self.next_layer is not None:
            return await self.next_layer.route_task(result, *args, **kwargs)
        
        return result
=======
