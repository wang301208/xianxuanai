import logging
import subprocess
from typing import ClassVar

from autogpt.core.ability.base import Ability, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.core.workspace import Workspace


class LintCode(Ability):
    """Run static analysis (lint) on a Python file."""

    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.LintCode",
        ),
        workspace_required=True,
    )

    def __init__(
        self,
        logger: logging.Logger,
        configuration: AbilityConfiguration,
        workspace: Workspace,
    ) -> None:
        self._logger = logger
        self._configuration = configuration
        self._workspace = workspace

    description: ClassVar[str] = "Run static code analysis on a file using Ruff or Flake8."

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "file_path": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Relative path to the Python file to lint.",
            required=True,
        )
    }

    async def __call__(self, file_path: str) -> AbilityResult:
        try:
            target_path = self._workspace.get_path(file_path)
        except ValueError as e:
            return AbilityResult(
                ability_name=self.name(),
                ability_args={"file_path": file_path},
                success=False,
                message=str(e),
            )

        lint_cmd = ["ruff", "check", str(target_path)]
        try:
            lint_proc = subprocess.run(
                lint_cmd, capture_output=True, text=True, check=False
            )
        except FileNotFoundError:
            lint_proc = subprocess.run(
                ["flake8", str(target_path)],
                capture_output=True,
                text=True,
                check=False,
            )
        lint_output = (lint_proc.stdout + lint_proc.stderr).strip()
        success = lint_proc.returncode == 0
        if success and not lint_output:
            lint_output = "ruff: no issues found"

        return AbilityResult(
            ability_name=self.name(),
            ability_args={"file_path": file_path},
            success=success,
            message=lint_output,
        )
