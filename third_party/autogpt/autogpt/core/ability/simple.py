import logging
from typing import ClassVar

import inflection

from autogpt.core.ability.base import Ability, AbilityConfiguration, AbilityRegistry
from autogpt.core.ability.builtins import BUILTIN_ABILITIES
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.configuration import Configurable, SystemConfiguration, SystemSettings
from autogpt.core.memory.base import Memory
from autogpt.core.plugin.base import PluginLocation, PluginStorageFormat
from autogpt.core.plugin.simple import SimplePluginService
from autogpt.core.resource.model_providers.schema import (
    ChatModelProvider,
    CompletionModelFunction,
    ModelProviderName,
)
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.core.workspace.base import Workspace
from autogpt.core.multimodal import MultimodalInput, embed_multimodal_input
from modules.deps import ModernDependencyManager


class SelfAssess(Ability):
    """Ability that checks recent memory entries for logical consistency."""

    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.simple.SelfAssess",
        ),
        memory_provider_required=True,
    )

    def __init__(
        self,
        logger: logging.Logger,
        configuration: AbilityConfiguration,
        memory: Memory,
    ):
        self._logger = logger
        self._configuration = configuration
        self._memory = memory

    description: ClassVar[str] = (
        "Review the last N memory entries for logical consistency."
    )

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "limit": JSONSchema(
            type=JSONSchema.Type.INTEGER,
            description=(
                "Number of recent memory entries to review for contradictions."
            ),
        )
    }

    async def __call__(self, limit: int = 5) -> AbilityResult:
        entries = self._memory.get(limit=limit)
        positive: dict[str, str] = {}
        negative: dict[str, str] = {}
        inconsistencies: list[tuple[str, str]] = []

        for entry in entries:
            normalized = entry.strip().lower()
            if normalized.startswith("not "):
                core = normalized[4:].strip()
                negative[core] = entry
                if core in positive:
                    inconsistencies.append((entry, positive[core]))
            else:
                core = normalized
                positive[core] = entry
                if core in negative:
                    inconsistencies.append((entry, negative[core]))

        if inconsistencies:
            details = "; ".join([f"'{a}' vs '{b}'" for a, b in inconsistencies])
            message = f"Potential inconsistencies found: {details}"
        else:
            message = f"No obvious inconsistencies found in last {limit} entries."

        return AbilityResult(
            ability_name=self.name(),
            ability_args={"limit": limit},
            success=True,
            message=message,
        )


# Register ability so it's available by default
BUILTIN_ABILITIES[SelfAssess.name()] = SelfAssess


class AbilityRegistryConfiguration(SystemConfiguration):
    """Configuration for the AbilityRegistry subsystem."""

    abilities: dict[str, AbilityConfiguration]


class AbilityRegistrySettings(SystemSettings):
    configuration: AbilityRegistryConfiguration


class SimpleAbilityRegistry(AbilityRegistry, Configurable):
    default_settings = AbilityRegistrySettings(
        name="simple_ability_registry",
        description="A simple ability registry.",
        configuration=AbilityRegistryConfiguration(
            abilities={
                ability_name: ability.default_configuration
                for ability_name, ability in BUILTIN_ABILITIES.items()
            },
        ),
    )

    def __init__(
        self,
        settings: AbilityRegistrySettings,
        logger: logging.Logger,
        memory: Memory,
        workspace: Workspace,
        model_providers: dict[ModelProviderName, ChatModelProvider],
    ):
        self._configuration = settings.configuration
        self._logger = logger
        self._memory = memory
        self._workspace = workspace
        self._model_providers = model_providers
        self._dependency_manager = ModernDependencyManager(logger)
        self._abilities: list[Ability] = []
        for (
            ability_name,
            ability_configuration,
        ) in self._configuration.abilities.items():
            self.register_ability(ability_name, ability_configuration)

    def register_ability(
        self, ability_name: str, ability_configuration: AbilityConfiguration
    ) -> None:
        ability_class = SimplePluginService.get_plugin(ability_configuration.location)
        ability_args = {
            "logger": self._logger.getChild(ability_name),
            "configuration": ability_configuration,
        }
        if ability_configuration.packages_required:
            self._dependency_manager.ensure_all(
                ability_configuration.packages_required
            )
        if ability_configuration.memory_provider_required:
            ability_args["memory"] = self._memory
        if ability_configuration.workspace_required:
            ability_args["workspace"] = self._workspace
        if ability_configuration.language_model_required:
            provider_name = ability_configuration.language_model_required.provider_name
            provider = self._model_providers.get(provider_name)
            if provider is None and hasattr(provider_name, "value"):
                provider = self._model_providers.get(provider_name.value)
            if provider is None:
                self._logger.debug(
                    "Skipping ability '%s' due to missing provider '%s'",
                    ability_name,
                    provider_name,
                )
                return
            ability_args["language_model_provider"] = provider
        ability = ability_class(**ability_args)
        self._abilities.append(ability)

    def list_abilities(self) -> list[str]:
        return [
            f"{ability.name()}: {ability.description}" for ability in self._abilities
        ]

    def dump_abilities(self) -> list[CompletionModelFunction]:
        return [ability.spec for ability in self._abilities]

    def get_ability(self, ability_name: str) -> Ability:
        for ability in self._abilities:
            if ability.name() == ability_name:
                return ability
        raise ValueError(f"Ability '{ability_name}' not found.")

    async def perform(
        self,
        ability_name: str,
        multimodal_input: MultimodalInput | None = None,
        **kwargs,
    ) -> AbilityResult:
        ability = self.get_ability(ability_name)
        if multimodal_input is not None:
            def _embed(data: str | bytes) -> list[float]:
                if isinstance(data, (bytes, bytearray)):
                    return [float(b) for b in data]
                return [float(ord(c)) for c in str(data)]

            kwargs["embedding"] = embed_multimodal_input(multimodal_input, _embed)
        try:
            return await ability(**kwargs)
        except Exception as err:
            self._logger.exception("Error performing ability '%s'", ability_name)
            return AbilityResult(
                ability_name=ability_name,
                ability_args={k: str(v) for k, v in kwargs.items()},
                success=False,
                message=str(err),
            )

    def optimize_ability(self, ability_name: str, metrics: dict[str, float]) -> None:
        current_config = self._configuration.abilities.get(ability_name)
        if not current_config:
            return
        optimized_route = (
            f"skills.{ability_name}_optimized.{inflection.camelize(ability_name)}"
        )
        location = PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route=optimized_route,
        )
        try:
            SimplePluginService.get_plugin(location)
        except Exception:
            self._logger.debug(
                "No optimized ability found for %s", ability_name
            )
            return
        new_config = current_config.copy(update={"location": location})
        self._abilities = [ab for ab in self._abilities if ab.name() != ability_name]
        self._configuration.abilities[ability_name] = new_config
        self.register_ability(ability_name, new_config)
        self._logger.info(
            "Replaced ability '%s' with optimized variant", ability_name
        )
