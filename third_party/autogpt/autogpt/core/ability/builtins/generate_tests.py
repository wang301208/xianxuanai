import logging
from typing import ClassVar

from autogpt.core.ability.base import Ability, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.planning.simple import LanguageModelConfiguration
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
from autogpt.core.resource.model_providers import (
    ChatMessage,
    ChatModelProvider,
    ModelProviderName,
    OpenAIModelName,
)
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.core.workspace import Workspace


class GenerateTests(Ability):
    """Generate pytest tests for a given Python file."""

    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.GenerateTests",
        ),
        language_model_required=LanguageModelConfiguration(
            model_name=OpenAIModelName.GPT3,
            provider_name=ModelProviderName.OPENAI,
            temperature=0.0,
        ),
        workspace_required=True,
    )

    def __init__(
        self,
        logger: logging.Logger,
        configuration: AbilityConfiguration,
        language_model_provider: ChatModelProvider,
        workspace: Workspace,
    ) -> None:
        self._logger = logger
        self._configuration = configuration
        self._language_model_provider = language_model_provider
        self._workspace = workspace

    description: ClassVar[str] = (
        "Generate example pytest tests for a Python module located at the given path."
    )

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "file_path": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Relative path to the Python file to generate tests for.",
            required=True,
        )
    }

    async def __call__(self, file_path: str) -> AbilityResult:
        try:
            target_path = self._workspace.get_path(file_path)
            if not target_path.exists():
                raise FileNotFoundError(f"File {file_path} does not exist")
            source = target_path.read_text()
        except Exception as e:  # pragma: no cover - best effort
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"file_path": file_path},
                success=False,
                message=str(e),
            )

        prompt = (
            "Write pytest unit tests for the following Python code. "
            "Return only the test code.\n\n" + source
        )
        model_response = await self._language_model_provider.create_chat_completion(
            model_prompt=[ChatMessage.user(prompt)],
            functions=[],
            model_name=self._configuration.language_model_required.model_name,
        )
        test_code = model_response.response.content or ""
        return AbilityResult(
            ability_name=self.name(),
            ability_args={"file_path": file_path},
            success=True,
            message=test_code,
        )
