from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import Field

from autogpt.core.agent.layered import LayeredAgent
from autogpt.core.configuration import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)
from autogpt.governance import Charter, load_charter


class GovernancePolicy(SystemConfiguration):
    """Settings that define governance rules for task routing."""

    charter_name: str = UserConfigurable(
        default="human_architect",
        description="Name of the charter file to load for governance rules.",
    )
    role: str = UserConfigurable(
        default="assistant",
        description="Role of the requesting agent within the charter.",
    )
    allowed_task_types: list[str] = UserConfigurable(
        default_factory=list,
        description="List of task types permitted by the governance layer.",
    )
    charter_path: Optional[Path] = None
    charter: Optional[Charter] = None


class GovernanceAgentSettings(SystemSettings):
    """System settings for the ``GovernanceAgent``."""

    policy: GovernancePolicy = Field(default_factory=GovernancePolicy)


class GovernanceAgent(LayeredAgent, Configurable[GovernanceAgentSettings]):
    """Top layer agent enforcing governance policy before delegating tasks."""

    default_settings = GovernanceAgentSettings(
        name="governance_agent",
        description="Routes tasks according to high level governance policy.",
    )

    def __init__(
        self,
        settings: GovernanceAgentSettings,
        next_layer: Optional[LayeredAgent] = None,
    ) -> None:
        super().__init__(next_layer=next_layer)
        self.settings = settings
        self.charter = load_charter(
            self.settings.policy.charter_name,
            directory=self.settings.policy.charter_path,
        )
        self.settings.policy.charter = self.charter

    def route_task(self, task: Any, role: Optional[str] = None, *args, **kwargs):
        """Route a task to the next layer if permitted by policy and charter."""

        role_name = role or self.settings.policy.role
        charter_role = next(
            (r for r in self.charter.roles if r.name == role_name), None
        )
        if charter_role is None:
            raise PermissionError(
                f"Role '{role_name}' is not defined in charter '{self.charter.name}'."
            )

        task_type = getattr(task, "type", getattr(task, "name", str(task)))

        is_core_change = getattr(task, "core_change", False) or task_type == "core_change"
        if is_core_change:
            if getattr(task, "approved_by", None) != "human_architect":
                raise PermissionError(
                    "Core architecture changes require approval from role 'human_architect'."
                )
            human_arch_role = next(
                (r for r in self.charter.roles if r.name == "human_architect"),
                None,
            )
            if human_arch_role is None or not any(
                p.name == "approve_core_change" for p in human_arch_role.permissions
            ):
                raise PermissionError(
                    "Charter is missing 'human_architect' role with 'approve_core_change' permission."
                )

        if charter_role.allowed_tasks and task_type not in charter_role.allowed_tasks:
            raise PermissionError(
                f"Role '{role_name}' is not permitted to perform task '{task_type}' "
                f"per charter '{self.charter.name}'."
            )

        allowed = self.settings.policy.allowed_task_types
        if allowed and task_type not in allowed:
            raise PermissionError(
                f"Task '{task_type}' is not permitted by the governance policy."
            )

        if self.next_layer is not None:
            return self.next_layer.route_task(task, *args, **kwargs)

        return task
