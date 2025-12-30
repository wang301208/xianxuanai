"""AutoGPT AI 配置文件模块。

本模块定义了 AI 代理的个性化配置，包括名称、角色、目标和预算等核心属性。
提供了配置的加载、保存和管理功能。

配置内容:
    - AI 名称和角色定义
    - 目标列表管理
    - API 预算控制
    - YAML 格式的持久化

设计特点:
    - 使用 Pydantic 进行数据验证
    - 支持 YAML 格式的配置文件
    - 提供默认值和类型安全
    - 灵活的目标格式处理
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AIProfile(BaseModel):
    """AI 代理的个性化配置模型。

    该类定义了 AI 代理的核心属性，包括身份、角色、目标和预算限制。
    使用 Pydantic 提供数据验证和序列化功能。

    属性说明:
        ai_name: AI 代理的名称，用于标识和交互
        ai_role: AI 代理的角色描述，定义其职责和能力范围
        ai_goals: AI 代理需要完成的目标列表
        api_budget: API 调用的最大预算（美元），0.0 表示无限制

    配置管理:
        - 支持从 YAML 文件加载配置
        - 支持将配置保存到 YAML 文件
        - 提供合理的默认值
        - 自动处理各种目标格式

    使用场景:
        - 代理个性化定制
        - 任务目标管理
        - 成本控制
        - 配置持久化
    """

    ai_name: str = ""  # AI 代理名称
    ai_role: str = ""  # AI 代理角色描述
    ai_goals: list[str] = Field(default_factory=list[str])  # 目标列表
    api_budget: float = 0.0  # API 预算限制（美元）

    @staticmethod
    def load(ai_settings_file: str | Path) -> "AIProfile":
        """
        Returns class object with parameters (ai_name, ai_role, ai_goals, api_budget)
        loaded from yaml file if it exists, else returns class with no parameters.

        Parameters:
            ai_settings_file (Path): The path to the config yaml file.

        Returns:
            cls (object): An instance of given cls object
        """

        try:
            with open(ai_settings_file, encoding="utf-8") as file:
                config_params = yaml.load(file, Loader=yaml.SafeLoader) or {}
        except FileNotFoundError:
            config_params = {}

        ai_name = config_params.get("ai_name", "")
        ai_role = config_params.get("ai_role", "")
        ai_goals = [
            str(goal).strip("{}").replace("'", "").replace('"', "")
            if isinstance(goal, dict)
            else str(goal)
            for goal in config_params.get("ai_goals", [])
        ]
        api_budget = config_params.get("api_budget", 0.0)

        return AIProfile(
            ai_name=ai_name, ai_role=ai_role, ai_goals=ai_goals, api_budget=api_budget
        )

    def save(self, ai_settings_file: str | Path) -> None:
        """
        Saves the class parameters to the specified file yaml file path as a yaml file.

        Parameters:
            ai_settings_file (Path): The path to the config yaml file.

        Returns:
            None
        """

        with open(ai_settings_file, "w", encoding="utf-8") as file:
            yaml.dump(self.dict(), file, allow_unicode=True)
