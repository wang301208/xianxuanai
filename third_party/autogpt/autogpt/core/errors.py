"""Core exception hierarchy for AutoGPT."""
from __future__ import annotations


class AutoGPTError(Exception):
    """Base class for all AutoGPT specific exceptions."""


class ConfigurationError(AutoGPTError):
    """Raised for configuration related issues."""


class PluginError(AutoGPTError):
    """Raised when a plugin operation fails."""


class EventBusError(AutoGPTError):
    """Raised for problems related to the event bus."""


class SkillSecurityError(AutoGPTError):
    """Raised when a skill fails security verification."""

    def __init__(self, skill: str, cause: str) -> None:
        super().__init__(f"Skill {skill} blocked: {cause}")
        self.skill = skill
        self.cause = cause


class SkillExecutionError(AutoGPTError):
    """Raised when a skill fails during execution."""

    def __init__(self, skill: str, cause: str) -> None:
        super().__init__(f"Skill {skill} failed: {cause}")
        self.skill = skill
        self.cause = cause
