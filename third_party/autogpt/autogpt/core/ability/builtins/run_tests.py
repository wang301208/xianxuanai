import asyncio
import logging
from typing import ClassVar

from autogpt.core.ability.base import Ability, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.core.workspace import Workspace


class RunTests(Ability):
    """Ability to execute the project's test suite."""

    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.RunTests",
        ),
        workspace_required=True,
    )

    def __init__(self, logger: logging.Logger, workspace: Workspace) -> None:
        self._logger = logger
        self._workspace = workspace

    description: ClassVar[str] = "Run tests in the workspace using pytest."

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "path": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Relative path from the workspace root to run tests in.",
            required=False,
        )
    }

    async def __call__(self, path: str = ".") -> AbilityResult:
        test_path = self._workspace.get_path(path)
        try:
            proc = await asyncio.create_subprocess_exec(
                "pytest",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(test_path),
            )
            stdout, _ = await proc.communicate()
            output = stdout.decode()
            success = proc.returncode == 0
            message = output
        except FileNotFoundError:
            success = False
            message = "pytest is not installed."
        except Exception as e:  # pragma: no cover - best effort
            success = False
            message = str(e)

        return AbilityResult(
            ability_name=self.name(),
            ability_args={"path": path},
            success=success,
            message=message,
        )

