"""AutoGPT agent base class definitions.

AutoGPT 代理基类定义。

This module defines the abstract base class that all AutoGPT agents must implement and documents the core interfaces.
本模块定义所有 AutoGPT 代理需实现的抽象基类，并说明其核心接口。
Agents are the central component responsible for executing tasks, making decisions, and managing workflows.
代理是 AutoGPT 系统的核心组件，负责执行任务、做出决策与管理工作流程。
"""

import abc
import logging
from pathlib import Path


class Agent(abc.ABC):
    """Abstract base class for AutoGPT agents.

    AutoGPT 代理的抽象基类。

    All concrete agent implementations must inherit from this class and provide the abstract methods.
    所有具体代理实现都必须继承此类并实现抽象方法。
    Agents autonomously execute tasks, make decisions, and orchestrate capability calls.
    代理负责自主执行任务、制定决策并调度能力调用。

    Core responsibilities / 核心职责:
        - Task planning and execution / 任务规划与执行
        - Capability selection and invocation / 能力选择与调用
        - State management and persistence / 状态管理与持久化
        - Interaction with external systems / 与外部系统交互

    Design pattern / 设计模式:
        Using an abstract base class enforces a unified interface that keeps the system extensible and maintainable.
        使用抽象基类确保接口统一，从而提升系统的可扩展性与可维护性。
    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialise an agent instance.

        初始化代理实例。

        Args / 参数:
            *args: Positional arguments defined by the subclass. / 由子类定义的位置参数。
            **kwargs: Keyword arguments defined by the subclass. / 由子类定义的关键字参数。

        Notes / 注意:
            Subclasses must implement this method to load configuration, initialise state, and prepare resources.
            子类必须实现此方法以完成配置加载、状态初始化与资源准备。
        """
        ...

    @classmethod
    @abc.abstractmethod
    def from_workspace(
        cls,
        workspace_path: Path,
        logger: logging.Logger,
    ) -> "Agent":
        """Construct an agent from a workspace.

        从工作空间创建代理实例。

        This factory method restores an agent from the configuration, state, and artefacts stored under the given path.
        该工厂方法会根据指定路径下的配置、状态与相关文件恢复代理。

        Args / 参数:
            workspace_path: Path to the workspace directory. / 工作空间目录路径。
            logger: Logger instance used by the agent. / 代理使用的日志记录器实例。

        Returns / 返回:
            Agent: The instantiated agent. / 创建完成的代理实例。

        Raises / 异常:
            FileNotFoundError: Raised when the workspace path does not exist. / 当工作空间路径不存在时抛出。
            ValueError: Raised when the workspace configuration is invalid. / 当工作空间配置无效时抛出。

        Notes / 注意:
            Subclasses must implement this to support state recovery from persisted workspaces.
            子类必须实现此方法以支持从持久化的工作空间恢复状态。
        """
        ...

    @abc.abstractmethod
    async def determine_next_ability(self, *args, **kwargs):
        """Determine the next capability to execute.

        确定下一个要执行的能力。

        This core decision method inspects the current state and goals to choose the capability that best advances the task.
        该核心决策方法会分析当前状态与目标，选择最能推进任务的能力。

        Args / 参数:
            *args: Positional inputs defined by the subclass. / 由子类定义的位置输入。
            **kwargs: Keyword inputs defined by the subclass. / 由子类定义的关键字输入。

        Returns / 返回:
            The subclass-defined structure describing the chosen capability and parameters.
            由子类定义的结构，描述所选能力及其参数。

        Notes / 注意:
            This is an asynchronous method because decision making may involve network calls, file I/O, or other long-running work.
            该方法为异步方法，因为决策过程可能包含网络请求、文件 I/O 或其他耗时操作。
        """
        ...

    @abc.abstractmethod
    def __repr__(self):
        """Return the string representation of the agent.

        返回代理的字符串表示。

        Returns / 返回:
            str: Descriptive information for debugging and logging. / 用于调试与日志的描述性字符串。

        Notes / 注意:
            Include key agent details such as type, state, and configuration.
            应包含代理类型、状态与配置等关键信息。
        """
        ...
