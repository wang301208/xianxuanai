"""Metadata schema and constants for meta-skills."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


class MetaSkillMetadata(TypedDict):
    """Typed metadata schema for meta-skills."""

    name: str
    version: str
    description: str
    protected: bool


# Common constant for the strategist's evolution template
META_SKILL_STRATEGY_EVOLUTION = "MetaSkill_StrategyEvolution_v1.0"

# Metadata flag indicating the skill should not be modified automatically
PROTECTED_FLAG = "protected"


@dataclass
class MetaSkill:
    """Simple representation of a meta-skill."""

    name: str
    version: str
    description: str
    protected: bool = False
