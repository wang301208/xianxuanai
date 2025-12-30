"""AutoGPT 代理能力层模块。

本模块实现了 AutoGPT 代理的能力选择和执行层，负责根据任务需求选择合适的能力，
并将执行结果传递给下一层或反馈处理器。

主要功能:
    - 能力注册表管理
    - 任务与能力的智能匹配
    - 执行计划的生成和路由
    - 性能反馈的收集和处理

设计模式:
    - 分层代理架构
    - 策略模式（能力选择）
    - 责任链模式（层级传递）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from autogpt.core.ability import AbilityRegistry, SimpleAbilityRegistry
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.agent.layered import LayeredAgent


class CapabilityAgent(LayeredAgent):
    """能力代理层，负责选择和委托执行任务能力。

    该层检查能力注册表中的可用能力，选择最匹配传入任务的能力。
    选中的能力被包装成执行计划并路由到下一层（通常是执行层）。
    执行后，性能数据可选择性地转发给反馈处理器，以便进化层学习结果。

    核心职责:
        - 能力发现和匹配
        - 执行计划生成
        - 结果路由和反馈
        - 性能数据收集

    架构特点:
        - 支持能力注册表的动态管理
        - 提供可选的反馈机制
        - 与其他代理层无缝集成
    """

    def __init__(
        self,
        ability_registry: AbilityRegistry | SimpleAbilityRegistry,
        next_layer: Optional[LayeredAgent] = None,
        feedback_handler: Optional[Callable[[str, AbilityResult], None]] = None,
    ) -> None:
        """初始化能力代理层。

        Args:
            ability_registry: 能力注册表，包含所有可用的能力
            next_layer: 下一层代理，通常是执行层
            feedback_handler: 可选的反馈处理器，用于收集性能数据

        注意:
            能力注册表是必需的，它定义了代理可以执行的所有操作。
            反馈处理器用于支持进化学习和性能优化。
        """
        super().__init__(next_layer=next_layer)
        self._ability_registry = ability_registry  # 能力注册表
        self._feedback_handler = feedback_handler  # 反馈处理器

    @classmethod
    def from_workspace(
        cls, workspace_path: Path, logger: Any
    ) -> "CapabilityAgent":  # pragma: no cover - simple passthrough
        raise NotImplementedError("CapabilityAgent does not support workspace loading")

    def __repr__(self) -> str:  # pragma: no cover - simple representation
        return f"CapabilityAgent(registry={self._ability_registry})"

    async def determine_next_ability(
        self, task: Any, *args: Any, **kwargs: Any
    ) -> tuple[dict[str, Any], AbilityResult]:
        """Select the best ability for ``task`` and delegate its execution."""

        ability_names = self._ability_registry.list_abilities()
        selected_ability = self._select_ability(task, ability_names)

        plan = {"next_ability": selected_ability, "ability_arguments": kwargs}

        if self.next_layer is not None:
            result = await self.next_layer.route_task(plan, *args, **kwargs)
        else:
            result = await self._ability_registry.perform(
                selected_ability, **plan["ability_arguments"]
            )

        if self._feedback_handler is not None:
            self._feedback_handler(selected_ability, result)

        return plan, result

    def _select_ability(
        self, task: Any, ability_names: Iterable[str]
    ) -> str:
        """Pick an ability that best matches ``task`` from ``ability_names``."""

        task_type = getattr(task, "type", getattr(task, "name", str(task)))
        for ability in ability_names:
            if task_type and task_type.lower() in ability.lower():
                return ability
        return next(iter(ability_names), "")

    def record_feedback(self, ability_name: str, result: AbilityResult) -> None:
        """Manually record feedback about an ability's performance."""

        if self._feedback_handler is not None:
            self._feedback_handler(ability_name, result)
