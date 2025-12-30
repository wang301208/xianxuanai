"""AutoGPT 代理看门狗功能模块。

本模块实现了代理的看门狗混入类，用于检测和处理代理的循环行为。
当检测到代理陷入重复执行相同命令的循环时，自动切换到更智能的语言模型重新思考。

主要功能:
    - 循环行为检测
    - 自动模型切换
    - 智能重新思考机制
    - 状态恢复和管理

设计模式:
    - 混入模式（Mixin Pattern）
    - 装饰器模式（行为增强）
    - 状态模式（模型切换）

使用场景:
    - 防止代理陷入无限循环
    - 提高问题解决效率
    - 优化资源使用
"""

from __future__ import annotations

import logging
from contextlib import ExitStack
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import BaseAgentConfiguration

from autogpt.models.action_history import EpisodicActionHistory

from ..base import BaseAgent

logger = logging.getLogger(__name__)


class WatchdogMixin:
    """代理看门狗混入类，为代理添加循环检测和智能切换功能。

    当代理开始循环执行相同操作时，看门狗会自动从快速语言模型切换到
    智能语言模型并重新思考问题，避免陷入无效的重复循环。

    核心机制:
        - 命令重复检测：监控连续执行的相同命令
        - 自动模型切换：从 FAST_LLM 切换到 SMART_LLM
        - 状态回滚：移除部分执行记录并重新开始
        - 智能重试：使用更强大的模型重新分析问题

    适用条件:
        - 必须应用于 BaseAgent 的派生类
        - 需要配置不同的快速和智能语言模型
        - 依赖事件历史记录进行循环检测

    性能优化:
        - 仅在检测到问题时才切换模型
        - 自动恢复到原始配置
        - 最小化对正常执行流程的影响
    """

    # 类型注解：必需的属性
    config: BaseAgentConfiguration  # 代理配置
    event_history: EpisodicActionHistory  # 事件历史记录

    def __init__(self, **kwargs) -> None:
        """初始化看门狗混入。

        Args:
            **kwargs: 传递给父类的关键字参数

        Raises:
            NotImplementedError: 如果不是应用于 BaseAgent 派生类

        注意:
            必须先初始化其他基类，因为需要从 BaseAgent 获取 event_history。
        """
        # 先初始化其他基类，因为需要从 BaseAgent 获取 event_history
        super(WatchdogMixin, self).__init__(**kwargs)

        # 验证混入只能应用于 BaseAgent 的派生类
        if not isinstance(self, BaseAgent):
            raise NotImplementedError(
                f"{__class__.__name__} can only be applied to BaseAgent derivatives"
            )

    async def propose_action(self, *args, **kwargs) -> BaseAgent.ThoughtProcessOutput:
        command_name, command_args, thoughts = await super(
            WatchdogMixin, self
        ).propose_action(*args, **kwargs)

        if not self.config.big_brain and self.config.fast_llm != self.config.smart_llm:
            previous_command, previous_command_args = None, None
            if len(self.event_history) > 1:
                # Detect repetitive commands
                previous_cycle = self.event_history.episodes[
                    self.event_history.cursor - 1
                ]
                previous_command = previous_cycle.action.name
                previous_command_args = previous_cycle.action.args

            rethink_reason = ""

            if not command_name:
                rethink_reason = "AI did not specify a command"
            elif (
                command_name == previous_command
                and command_args == previous_command_args
            ):
                rethink_reason = f"Repititive command detected ({command_name})"

            if rethink_reason:
                logger.info(f"{rethink_reason}, re-thinking with SMART_LLM...")
                with ExitStack() as stack:

                    @stack.callback
                    def restore_state() -> None:
                        # Executed after exiting the ExitStack context
                        self.config.big_brain = False

                    # Remove partial record of current cycle
                    self.event_history.rewind()

                    # Switch to SMART_LLM and re-think
                    self.big_brain = True
                    return await self.propose_action(*args, **kwargs)

        return command_name, command_args, thoughts
