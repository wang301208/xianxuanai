from __future__ import annotations

"""Global accessors for the dynamic skill registry."""

from pathlib import Path
from typing import Iterable, Optional

from modules.skills import SkillRegistry, SkillSpec, SkillRegistrationError

_GLOBAL_SKILL_REGISTRY: SkillRegistry | None = None


def get_skill_registry() -> SkillRegistry:
    global _GLOBAL_SKILL_REGISTRY
    if _GLOBAL_SKILL_REGISTRY is None:
        _GLOBAL_SKILL_REGISTRY = SkillRegistry()
    return _GLOBAL_SKILL_REGISTRY


def register_skill(spec: SkillSpec, handler=None, *, replace: bool = False) -> None:
    registry = get_skill_registry()
    registry.register(spec, handler=handler, replace=replace)


def unregister_skill(name: str) -> None:
    registry = get_skill_registry()
    registry.unregister(name)


def refresh_skills_from_directory(
    roots: Iterable[Path | str],
    *,
    pattern: str = "*.skill.json",
    prune_missing: bool = False,
) -> None:
    registry = get_skill_registry()
    for root in roots:
        registry.refresh_from_directory(root, pattern=pattern, prune_missing=prune_missing)


__all__ = [
    "get_skill_registry",
    "register_skill",
    "unregister_skill",
    "refresh_skills_from_directory",
    "SkillRegistry",
    "SkillSpec",
    "SkillRegistrationError",
]

