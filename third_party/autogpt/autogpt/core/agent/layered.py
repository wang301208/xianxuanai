from __future__ import annotations

from typing import Any, Optional

from autogpt.core.agent.base import Agent


class LayeredAgent(Agent):
    """支持分层任务处理的智能体实现。

    该类实现了责任链模式，任务可以通过多个处理层传递。每一层可以处理
    任务的特定方面，然后将其转发到链中的下一层。

    分层架构的优势：
    - 不同处理层之间的关注点分离
    - 智能体能力的灵活组合
    - 处理管道的易扩展和修改
    - 通过层特定的错误处理实现容错

    属性:
        next_layer: 处理链中的下一个智能体层，如果这是最后一层则为None

    示例:
        >>> planning_layer = PlanningAgent()
        >>> execution_layer = ExecutionAgent(next_layer=planning_layer)
        >>> result = execution_layer.route_task(task)
    """

    next_layer: Optional["LayeredAgent"]

    def __init__(self, *args, next_layer: Optional["LayeredAgent"] = None, **kwargs):
        """初始化分层智能体。

        参数:
            *args: 传递给父类Agent的位置参数
            next_layer: 处理链中的下一层，如果是最后一层则为None
            **kwargs: 传递给父类Agent的关键字参数
        """
        super().__init__(*args, **kwargs)
        self.next_layer = next_layer

    def route_task(self, task: Any, *args, **kwargs):
        """将任务路由到处理链中的下一层。

        该方法实现了默认的路由行为，即简单地将任务转发到下一层。
        子类应该重写此方法，在转发之前实现层特定的处理。

        参数:
            task: 要处理和路由的任务
            *args: 任务处理的额外位置参数
            **kwargs: 任务处理的额外关键字参数

        返回:
            下一层处理的结果

        异常:
            NotImplementedError: 如果没有下一层来路由任务。
                                这通常表示配置错误或该层应该自己处理任务。
        """
        if self.next_layer is not None:
            return self.next_layer.route_task(task, *args, **kwargs)
        raise NotImplementedError("No next layer to route task to.")
