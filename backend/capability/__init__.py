from .module_registry import (
    available_modules,
    combine_modules,
    disabled_modules,
    disable_module,
    enable_module,
    get_module,
    register_module,
    unregister_module,
    is_module_enabled,
)
from .runtime_loader import RuntimeModuleManager
from .module_factory import (
    EvolutionaryModuleFactory,
    ModuleBlueprint,
    ModuleSpec,
    EvolvedCapabilityModule,
    ModuleGrowthController,
)
try:  # Optional dependency tree; skip when auxiliary packages unavailable.
    from .skill_registry import (
        get_skill_registry,
        register_skill,
        unregister_skill,
        refresh_skills_from_directory,
        SkillRegistry,
        SkillSpec,
        SkillRegistrationError,
    )
except Exception:  # pragma: no cover - expose stubs for lightweight environments
    get_skill_registry = register_skill = unregister_skill = refresh_skills_from_directory = None  # type: ignore[assignment]
    SkillRegistry = SkillSpec = SkillRegistrationError = None  # type: ignore[assignment]

__all__ = [
    "available_modules",
    "combine_modules",
    "disabled_modules",
    "disable_module",
    "enable_module",
    "get_module",
    "register_module",
    "unregister_module",
    "is_module_enabled",
    "RuntimeModuleManager",
    "EvolutionaryModuleFactory",
    "ModuleBlueprint",
    "ModuleSpec",
    "EvolvedCapabilityModule",
    "ModuleGrowthController",
    "get_skill_registry",
    "register_skill",
    "unregister_skill",
    "refresh_skills_from_directory",
    "SkillRegistry",
    "SkillSpec",
    "SkillRegistrationError",
]
