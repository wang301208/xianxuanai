import importlib
import json
import logging
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import ClassVar

import inflection

from autogpt.core.ability.base import Ability, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.plugin.simple import (
    PluginLocation,
    PluginStorageFormat,
    SimplePluginService,
)
from autogpt.core.utils.json_schema import JSONSchema


class CreateNewAbility(Ability):
    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.CreateNewAbility",
        ),
    )

    def __init__(
        self,
        logger: logging.Logger,
        configuration: AbilityConfiguration,
    ):
        self._logger = logger
        self._configuration = configuration

    description: ClassVar[str] = "Create a new ability by writing python code."

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "ability_name": JSONSchema(
            description="A meaningful and concise name for the new ability.",
            type=JSONSchema.Type.STRING,
            required=True,
        ),
        "description": JSONSchema(
            description=(
                "A detailed description of the ability and its uses, "
                "including any limitations."
            ),
            type=JSONSchema.Type.STRING,
            required=True,
        ),
        "arguments": JSONSchema(
            description="A list of arguments that the ability will accept.",
            type=JSONSchema.Type.ARRAY,
            items=JSONSchema(
                type=JSONSchema.Type.OBJECT,
                properties={
                    "name": JSONSchema(
                        description="The name of the argument.",
                        type=JSONSchema.Type.STRING,
                    ),
                    "type": JSONSchema(
                        description=(
                            "The type of the argument. "
                            "Must be a standard json schema type."
                        ),
                        type=JSONSchema.Type.STRING,
                    ),
                    "description": JSONSchema(
                        description=(
                            "A detailed description of the argument and its uses."
                        ),
                        type=JSONSchema.Type.STRING,
                    ),
                },
            ),
        ),
        "required_arguments": JSONSchema(
            description="A list of the names of the arguments that are required.",
            type=JSONSchema.Type.ARRAY,
            items=JSONSchema(
                description="The names of the arguments that are required.",
                type=JSONSchema.Type.STRING,
            ),
        ),
        "package_requirements": JSONSchema(
            description=(
                "A list of the names of the Python packages that are required to "
                "execute the ability."
            ),
            type=JSONSchema.Type.ARRAY,
            items=JSONSchema(
                description=(
                    "The of the Python package that is required to execute the ability."
                ),
                type=JSONSchema.Type.STRING,
            ),
        ),
        "code": JSONSchema(
            description=(
                "The Python code that will be executed when the ability is called."
            ),
            type=JSONSchema.Type.STRING,
            required=True,
        ),
    }

    async def __call__(
        self,
        ability_name: str,
        description: str,
        arguments: list[dict],
        required_arguments: list[str],
        package_requirements: list[str],
        code: str,
    ) -> AbilityResult:
        module_name = inflection.underscore(ability_name)
        class_name = inflection.camelize(module_name)

        builtins_dir = Path(__file__).resolve().parent
        ability_file = builtins_dir / f"{module_name}.py"
        ability_file.parent.mkdir(parents=True, exist_ok=True)

        params_schema: dict[str, JSONSchema] = {}
        param_lines: list[str] = []
        for arg in arguments:
            arg_name = arg.get("name")
            arg_type = JSONSchema.Type(arg.get("type", "string"))
            arg_desc = arg.get("description")
            required = arg_name in required_arguments
            params_schema[arg_name] = JSONSchema(
                description=arg_desc, type=arg_type, required=required
            )
            param_lines.append(
                f'        "{arg_name}": JSONSchema(description={arg_desc!r}, '
                f"type=JSONSchema.Type.{arg_type.name}, required={required}),"
            )

        params_block = "\n".join(param_lines)

        ability_code = f"""import logging
from typing import ClassVar

from autogpt.core.ability.base import Ability, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
from autogpt.core.utils.json_schema import JSONSchema


class {class_name}(Ability):
    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.{module_name}.{class_name}",
        ),
        packages_required={package_requirements},
    )

    def __init__(
        self,
        logger: logging.Logger,
        configuration: AbilityConfiguration,
    ):
        self._logger = logger
        self._configuration = configuration

    description: ClassVar[str] = {description!r}

    parameters: ClassVar[dict[str, JSONSchema]] = {{
{params_block}
    }}

    async def __call__(self, **kwargs) -> AbilityResult:
{textwrap.indent(code, ' ' * 8)}
"""

        ability_file.write_text(ability_code)

        # Run static analysis on the generated ability
        lint_cmd = ["ruff", "check", str(ability_file)]
        try:
            lint_proc = subprocess.run(
                lint_cmd, capture_output=True, text=True, check=False
            )
        except FileNotFoundError:
            lint_proc = subprocess.run(
                ["flake8", str(ability_file)], capture_output=True, text=True, check=False
            )
        lint_output = (lint_proc.stdout + lint_proc.stderr).strip()

        if lint_proc.returncode != 0:
            if ability_file.exists():
                ability_file.unlink()
            return AbilityResult(
                ability_name=ability_name,
                ability_args={
                    "ability_name": ability_name,
                    "description": description,
                    "parameters": json.dumps(
                        {k: v.to_dict() for k, v in params_schema.items()}
                    ),
                },
                success=False,
                message=f"Static analysis failed for {ability_name}:\n{lint_output}",
            )

        if package_requirements:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", *package_requirements],
                check=False,
            )

        importlib.invalidate_caches()
        plugin_location = PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route=f"autogpt.core.ability.builtins.{module_name}.{class_name}",
        )
        ability_class = SimplePluginService.get_plugin(plugin_location)

        from autogpt.core.ability.builtins import BUILTIN_ABILITIES

        BUILTIN_ABILITIES[ability_name] = ability_class

        if not lint_output:
            lint_output = "ruff: no issues found"

        return AbilityResult(
            ability_name=ability_name,
            ability_args={
                "ability_name": ability_name,
                "description": description,
                "parameters": json.dumps(
                    {k: v.to_dict() for k, v in params_schema.items()}
                ),
            },
            success=True,
            message=f"Ability {ability_name} created.\n{lint_output}",
        )
