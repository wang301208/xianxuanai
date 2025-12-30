import inspect
import logging
import sys
from pathlib import Path

import inflection
import pytest

from autogpt.core.ability.builtins import BUILTIN_ABILITIES
from autogpt.core.ability.builtins.create_new_ability import CreateNewAbility
from autogpt.core.ability.simple import (
    AbilityRegistryConfiguration,
    AbilityRegistrySettings,
    SimpleAbilityRegistry,
)
from autogpt.core.memory.simple import SimpleMemory
from autogpt.core.workspace.simple import SimpleWorkspace


@pytest.mark.asyncio
async def test_create_and_execute_new_ability(tmp_path):
    logger = logging.getLogger("test")

    # Set up workspace and memory
    workspace_settings = SimpleWorkspace.default_settings.copy(deep=True)
    workspace_settings.configuration.root = str(tmp_path)
    workspace = SimpleWorkspace(workspace_settings, logger)
    memory = SimpleMemory(SimpleMemory.default_settings, logger, workspace)

    creator = CreateNewAbility(logger, CreateNewAbility.default_configuration)

    ability_name = "TestAbility"
    message = "Hello from new ability!"
    code = (
        "return AbilityResult(ability_name='TestAbility', ability_args={}, "
        "success=True, message='" + message + "')"
    )

    result = await creator(
        ability_name=ability_name,
        description="A simple test ability",
        arguments=[],
        required_arguments=[],
        package_requirements=[],
        code=code,
    )
    assert result.success

    ability_class = BUILTIN_ABILITIES[ability_name]
    ability_key = ability_class.name()
    registry_settings = AbilityRegistrySettings(
        name="registry",
        description="",
        configuration=AbilityRegistryConfiguration(
            abilities={ability_key: ability_class.default_configuration}
        ),
    )
    registry = SimpleAbilityRegistry(registry_settings, logger, memory, workspace, {})

    try:
        exec_result = await registry.perform(ability_key)
        assert exec_result.message == message
    finally:
        # Clean up generated ability
        builtins_dir = Path(inspect.getfile(CreateNewAbility)).resolve().parent
        module_name = inflection.underscore(ability_name)
        ability_file = builtins_dir / f"{module_name}.py"
        if ability_file.exists():
            ability_file.unlink()
        cache_dir = builtins_dir / "__pycache__"
        if cache_dir.exists():
            for pyc in cache_dir.glob(f"{module_name}*.pyc"):
                pyc.unlink()
        BUILTIN_ABILITIES.pop(ability_name, None)
        sys.modules.pop(f"autogpt.core.ability.builtins.{module_name}", None)


@pytest.mark.asyncio
async def test_create_new_ability_static_analysis_fail():
    logger = logging.getLogger("test")
    creator = CreateNewAbility(logger, CreateNewAbility.default_configuration)

    ability_name = "BadAbility"
    # Unused import should trigger a lint error
    code = (
        "import os\n"
        "return AbilityResult(ability_name='BadAbility', ability_args={}, "
        "success=True, message='hi')"
    )

    result = await creator(
        ability_name=ability_name,
        description="An invalid ability",
        arguments=[],
        required_arguments=[],
        package_requirements=[],
        code=code,
    )

    assert not result.success
    assert ability_name not in BUILTIN_ABILITIES
    builtins_dir = Path(inspect.getfile(CreateNewAbility)).resolve().parent
    module_name = inflection.underscore(ability_name)
    assert not (builtins_dir / f"{module_name}.py").exists()
